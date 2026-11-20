package com.vocalsplitter.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

class SeparationService : Service() {

    companion object {
        const val ACTION_START = "com.vocalsplitter.app.START"
        const val ACTION_CANCEL = "com.vocalsplitter.app.CANCEL"
        private const val EXTRA_URI = "uri"
        private const val EXTRA_NAME = "name"
        private const val EXTRA_MODE = "mode"
        private const val CHANNEL_ID = "separation"
        private const val NOTIF_ID = 1
        private const val DONE_ID = 2
        private const val MODEL_ASSET = "models/UVR-MDX-NET-Voc_FT.onnx"

        fun start(ctx: Context, uri: Uri, name: String, mode: Mode) {
            val i = Intent(ctx, SeparationService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_URI, uri.toString())
                .putExtra(EXTRA_NAME, name)
                .putExtra(EXTRA_MODE, mode.name)
            ContextCompat.startForegroundService(ctx, i)
        }

        fun cancel(ctx: Context) {
            ctx.startService(Intent(ctx, SeparationService::class.java).setAction(ACTION_CANCEL))
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null
    private var wake: PowerManager.WakeLock? = null
    private var lastNotif = 0L
    private lateinit var nm: NotificationManager

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Separation progress", NotificationManager.IMPORTANCE_LOW)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL -> {
                val j = job
                if (j != null && j.isActive) {
                    SeparationState.ui.value = SeparationState.ui.value.copy(message = "Cancelling…")
                    j.cancel()
                } else {
                    stopSelf()
                }
            }

            ACTION_START -> {
                ServiceCompat.startForeground(
                    this, NOTIF_ID, buildNotification("Preparing…", 0),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                )
                if (job?.isActive == true) return START_NOT_STICKY
                val uri = Uri.parse(intent.getStringExtra(EXTRA_URI) ?: "")
                val name = intent.getStringExtra(EXTRA_NAME) ?: "song"
                val mode = try {
                    Mode.valueOf(intent.getStringExtra(EXTRA_MODE) ?: Mode.BEST.name)
                } catch (_: Exception) {
                    Mode.BEST
                }
                launchJob(uri, name, mode)
            }
        }
        return START_NOT_STICKY
    }

    private fun launchJob(uri: Uri, name: String, mode: Mode) {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "VocalSplitter::separation").apply {
            setReferenceCounted(false)
            acquire(8L * 60 * 60 * 1000)
        }
        job = scope.launch {
            val active = { isActive }
            try {
                process(uri, name, mode, active)
            } catch (e: CancellationException) {
                SeparationState.clear(applicationContext)
                SeparationState.ui.value = UiState(Phase.Idle, message = "Cancelled.")
                throw e
            } catch (t: Throwable) {
                val msg = if (t is OutOfMemoryError) {
                    "Out of memory. Close other apps and try again, or use a shorter song."
                } else {
                    t.message ?: t.javaClass.simpleName
                }
                SeparationState.clear(applicationContext)
                SeparationState.ui.value = UiState(Phase.Error, error = msg, message = "Failed")
            } finally {
                wake?.let { if (it.isHeld) it.release() }
                ServiceCompat.stopForeground(this@SeparationService, ServiceCompat.STOP_FOREGROUND_REMOVE)
                when (SeparationState.ui.value.phase) {
                    Phase.Done -> postFinal("Finished", "Open the app to mix and download.")
                    Phase.Error -> postFinal("Separation failed", SeparationState.ui.value.error ?: "")
                    else -> {}
                }
                stopSelf()
            }
        }
    }

    private fun process(uri: Uri, name: String, mode: Mode, active: () -> Boolean) {
        val ctx = applicationContext
        SeparationState.clear(ctx)
        publish(0f, "Reading audio file…")

        val audio = AudioDecoder.decode(ctx, uri, { publish(it * 0.04f, "Reading audio file…") }, active)
        var left = audio.left
        var right = audio.right
        if (left.isEmpty()) throw IllegalStateException("The audio file is empty or not supported.")
        if (audio.sampleRate != 44100) {
            publish(0.04f, "Converting sample rate…")
            left = Resampler.resample(left, audio.sampleRate, 44100)
            right = Resampler.resample(right, audio.sampleRate, 44100)
        }

        publish(0.05f, "Loading AI model…")
        val modelPath = prepareModel(ctx)

        val cores = Runtime.getRuntime().availableProcessors()
        val threads = if (mode == Mode.SUPER) cores.coerceIn(2, 8) else cores.coerceIn(2, 4)
        val denoise = mode == Mode.BEST

        MdxSeparator(modelPath, threads).use { sep ->
            val t0 = SystemClock.elapsedRealtime()
            val stems = sep.separate(left, right, denoise, { done, total ->
                val elapsed = (SystemClock.elapsedRealtime() - t0) / 1000.0
                val eta = if (done > 0) (elapsed / done * (total - done)).toLong() else -1L
                publish(
                    0.06f + 0.90f * (done / total.toFloat()),
                    "Separating… part $done of $total",
                    eta,
                )
            }, active)

            publish(0.97f, "Saving results…")
            StemIO.write(SeparationState.vocalsFile(ctx), stems[0], stems[1], null, null)
            StemIO.write(SeparationState.instFile(ctx), left, right, stems[0], stems[1])
            val result = StemResult(name, left.size.toLong())
            SeparationState.saveMeta(ctx, result)
            SeparationState.ui.value = UiState(Phase.Done, 1f, "Done", -1L, result)
        }
    }

    private fun prepareModel(ctx: Context): String {
        val target = File(ctx.filesDir, "model.onnx")
        val expected = try {
            ctx.assets.openFd(MODEL_ASSET).use { it.length }
        } catch (e: Exception) {
            -1L
        }
        if (!target.exists() || (expected > 0 && target.length() != expected)) {
            val tmp = File(ctx.filesDir, "model.onnx.tmp")
            ctx.assets.open(MODEL_ASSET).use { input ->
                FileOutputStream(tmp).use { output -> input.copyTo(output, 1 shl 20) }
            }
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) throw IllegalStateException("Could not prepare the AI model.")
        }
        return target.absolutePath
    }

    private fun publish(progress: Float, msg: String, eta: Long = -1L) {
        val p = progress.coerceIn(0f, 1f)
        SeparationState.ui.value = UiState(Phase.Working, p, msg, eta)
        val now = SystemClock.elapsedRealtime()
        if (now - lastNotif > 1500) {
            lastNotif = now
            nm.notify(NOTIF_ID, buildNotification(msg, (p * 100).toInt()))
        }
    }

    private fun openAppIntent(): PendingIntent = PendingIntent.getActivity(
        this, 0,
        Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
    )

    private fun buildNotification(text: String, pct: Int): Notification {
        val cancel = PendingIntent.getService(
            this, 1,
            Intent(this, SeparationService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("Separating vocals")
            .setContentText(text)
            .setProgress(100, pct, pct == 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openAppIntent())
            .addAction(0, "Cancel", cancel)
            .build()
    }

    private fun postFinal(title: String, text: String) {
        val n = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle(title)
            .setContentText(text)
            .setAutoCancel(true)
            .setContentIntent(openAppIntent())
            .build()
        nm.notify(DONE_ID, n)
    }

    override fun onDestroy() {
        wake?.let { if (it.isHeld) it.release() }
        scope.cancel()
        super.onDestroy()
    }
}
