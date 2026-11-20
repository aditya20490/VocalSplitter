package com.vocalsplitter.app

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/** Live preview: mixes the two stem files with the current slider gains while playing. */
class MixPlayer(
    private val vocals: File,
    private val inst: File,
    val totalFrames: Long,
) {
    @Volatile var vocalGain = 1f
    @Volatile var instGain = 1f
    @Volatile var positionFrames = 0L
    @Volatile var isPlaying = false
        private set
    @Volatile private var seekTarget = -1L
    private var thread: Thread? = null

    private companion object {
        const val SR = 44100
        const val BLOCK = 2048
    }

    @Synchronized
    fun play() {
        if (isPlaying) return
        if (positionFrames >= totalFrames) positionFrames = 0
        isPlaying = true
        val t = Thread({ loop() }, "mix-player")
        thread = t
        t.start()
    }

    @Synchronized
    fun pause() {
        isPlaying = false
        thread?.join()
        thread = null
    }

    fun seekTo(frame: Long) {
        val f = frame.coerceIn(0L, totalFrames)
        if (isPlaying) seekTarget = f else positionFrames = f
    }

    fun release() = pause()

    private fun loop() {
        var track: AudioTrack? = null
        var finished = false
        try {
            val minBuf = AudioTrack.getMinBufferSize(
                SR, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT
            )
            val bufBytes = max(minBuf, BLOCK * 8 * 3)
            val t = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(SR)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build()
                )
                .setBufferSizeInBytes(bufBytes)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            track = t
            t.play()

            RandomAccessFile(vocals, "r").use { v ->
                RandomAccessFile(inst, "r").use { i ->
                    val vb = ByteArray(BLOCK * 8)
                    val ib = ByteArray(BLOCK * 8)
                    val vf = FloatArray(BLOCK * 2)
                    val iff = FloatArray(BLOCK * 2)
                    val out = FloatArray(BLOCK * 2)
                    var cursor = positionFrames
                    var prevVg = vocalGain
                    var prevIg = instGain

                    while (isPlaying) {
                        val s = seekTarget
                        if (s >= 0) {
                            cursor = s
                            seekTarget = -1L
                            positionFrames = cursor
                            t.pause()
                            t.flush()
                            t.play()
                        }
                        if (cursor >= totalFrames) {
                            finished = true
                            break
                        }
                        val n = min(BLOCK.toLong(), totalFrames - cursor).toInt()
                        v.seek(cursor * 8)
                        v.readFully(vb, 0, n * 8)
                        i.seek(cursor * 8)
                        i.readFully(ib, 0, n * 8)
                        ByteBuffer.wrap(vb, 0, n * 8).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(vf, 0, n * 2)
                        ByteBuffer.wrap(ib, 0, n * 8).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(iff, 0, n * 2)

                        // Read the live slider gains for this block and ramp smoothly (no clicks).
                        val vg = vocalGain
                        val ig = instGain
                        for (f in 0 until n) {
                            val a = (f + 1).toFloat() / n
                            val gv = prevVg + (vg - prevVg) * a
                            val gi = prevIg + (ig - prevIg) * a
                            val j = f * 2
                            var l = vf[j] * gv + iff[j] * gi
                            var r = vf[j + 1] * gv + iff[j + 1] * gi
                            if (l > 1f) l = 1f else if (l < -1f) l = -1f
                            if (r > 1f) r = 1f else if (r < -1f) r = -1f
                            out[j] = l
                            out[j + 1] = r
                        }
                        prevVg = vg
                        prevIg = ig

                        var w = 0
                        while (w < n * 2 && isPlaying) {
                            val r = t.write(out, w, n * 2 - w, AudioTrack.WRITE_BLOCKING)
                            if (r < 0) throw IOException("Audio write error $r")
                            w += r
                        }
                        if (w >= n * 2) {
                            cursor += n
                            positionFrames = cursor
                        }
                    }
                }
            }
        } catch (_: Exception) {
        } finally {
            try {
                if (finished) {
                    track?.stop()
                    Thread.sleep(450)
                } else {
                    track?.pause()
                    track?.flush()
                }
            } catch (_: Exception) {
            }
            try { track?.release() } catch (_: Exception) {}
            if (finished) positionFrames = totalFrames
            isPlaying = false
        }
    }
}
