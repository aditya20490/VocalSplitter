package com.vocalsplitter.app

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlinx.coroutines.CancellationException
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

class DecodedAudio(val left: FloatArray, val right: FloatArray, val sampleRate: Int)

private class Growable(initial: Int) {
    var data = FloatArray(max(initial, 4096))
    var size = 0
    fun add(v: Float) {
        if (size == data.size) data = data.copyOf(data.size + data.size / 2 + 4096)
        data[size++] = v
    }
    fun toArray(): FloatArray = if (size == data.size) data else data.copyOf(size)
}

object AudioDecoder {
    fun decode(
        context: Context,
        uri: Uri,
        onProgress: (Float) -> Unit,
        isActive: () -> Boolean,
    ): DecodedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            var track = -1
            var found: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if ((f.getString(MediaFormat.KEY_MIME) ?: "").startsWith("audio/")) {
                    track = i
                    found = f
                    break
                }
            }
            val format = found ?: throw IllegalStateException("No audio track found in this file.")
            extractor.selectTrack(track)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val durationUs =
                if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
            var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var pcmFloat = false

            val estimate = ((durationUs / 1_000_000.0) * sampleRate * 1.02).toInt() + 8192
            val left = Growable(estimate)
            val right = Growable(estimate)

            val c = MediaCodec.createDecoderByType(mime)
            codec = c
            c.configure(format, null, null, 0)
            c.start()

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!isActive()) throw CancellationException("cancelled")

                if (!inputDone) {
                    val inIdx = c.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = c.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            c.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            c.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = c.dequeueOutputBuffer(info, 10_000)
                when {
                    outIdx >= 0 -> {
                        if (info.size > 0) {
                            val out = c.getOutputBuffer(outIdx)!!
                            out.position(info.offset)
                            out.limit(info.offset + info.size)
                            out.order(ByteOrder.LITTLE_ENDIAN)
                            val ch = max(channels, 1)
                            if (pcmFloat) {
                                val fb = out.asFloatBuffer()
                                val frames = fb.remaining() / ch
                                for (f in 0 until frames) {
                                    val l = fb.get(f * ch)
                                    val r = if (ch >= 2) fb.get(f * ch + 1) else l
                                    left.add(l)
                                    right.add(r)
                                }
                            } else {
                                val sb = out.asShortBuffer()
                                val frames = sb.remaining() / ch
                                for (f in 0 until frames) {
                                    val l = sb.get(f * ch) / 32768f
                                    val r = if (ch >= 2) sb.get(f * ch + 1) / 32768f else l
                                    left.add(l)
                                    right.add(r)
                                }
                            }
                            if (durationUs > 0) {
                                onProgress((info.presentationTimeUs / durationUs.toFloat()).coerceIn(0f, 1f))
                            }
                        }
                        c.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val nf = c.outputFormat
                        if (nf.containsKey(MediaFormat.KEY_SAMPLE_RATE)) sampleRate = nf.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        if (nf.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) channels = nf.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        pcmFloat = nf.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                            nf.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                    }
                }
            }
            return DecodedAudio(left.toArray(), right.toArray(), sampleRate)
        } finally {
            try { codec?.stop() } catch (_: Exception) {}
            try { codec?.release() } catch (_: Exception) {}
            extractor.release()
        }
    }
}

/** High-quality windowed-sinc resampler (used only when the song is not 44.1 kHz). */
object Resampler {
    fun resample(x: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate == dstRate) return x
        val ratio = dstRate.toDouble() / srcRate
        val outLen = (x.size * ratio).toInt()
        val out = FloatArray(outLen)
        val cutoff = if (ratio < 1.0) ratio else 1.0
        val halfWidth = 24.0 / cutoff
        for (i in 0 until outLen) {
            val t = i / ratio
            val lo = max(ceil(t - halfWidth).toInt(), 0)
            val hi = min(floor(t + halfWidth).toInt(), x.size - 1)
            var acc = 0.0
            for (j in lo..hi) {
                val d = j - t
                val s = d * cutoff
                val sinc = if (abs(s) < 1e-9) 1.0 else sin(PI * s) / (PI * s)
                val w = 0.5 + 0.5 * cos(PI * d / halfWidth)
                acc += x[j] * sinc * w * cutoff
            }
            out[i] = acc.toFloat()
        }
        return out
    }
}
