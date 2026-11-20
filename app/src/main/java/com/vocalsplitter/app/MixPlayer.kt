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
        const val BLOCK = 4096
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
            val bufBytes = max(minBuf, BLOCK * 8 * 4)
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

                        val vg = vocalGain
                        val ig = instGain
                        for (j in 0 until n * 2) {
                            var x = vf[j] * vg + iff[j] * ig
                            if (x > 1f) x = 1f else if (x < -1f) x = -1f
                            out[j] = x
                        }
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
