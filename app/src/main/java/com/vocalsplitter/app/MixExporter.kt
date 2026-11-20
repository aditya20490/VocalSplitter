package com.vocalsplitter.app

import android.content.ContentValues
import android.content.Context
import android.provider.MediaStore
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object MixExporter {
    private const val BLOCK = 16384
    private const val SR = 44100

    private fun forEachMixBlock(
        vocals: File, inst: File, totalFrames: Long, vg: Float, ig: Float,
        action: (FloatArray, Int) -> Unit,
    ) {
        RandomAccessFile(vocals, "r").use { v ->
            RandomAccessFile(inst, "r").use { i ->
                val vb = ByteArray(BLOCK * 8)
                val ib = ByteArray(BLOCK * 8)
                val vf = FloatArray(BLOCK * 2)
                val iff = FloatArray(BLOCK * 2)
                val mix = FloatArray(BLOCK * 2)
                var cursor = 0L
                while (cursor < totalFrames) {
                    val n = min(BLOCK.toLong(), totalFrames - cursor).toInt()
                    v.readFully(vb, 0, n * 8)
                    i.readFully(ib, 0, n * 8)
                    ByteBuffer.wrap(vb, 0, n * 8).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(vf, 0, n * 2)
                    ByteBuffer.wrap(ib, 0, n * 8).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(iff, 0, n * 2)
                    for (j in 0 until n * 2) mix[j] = vf[j] * vg + iff[j] * ig
                    action(mix, n * 2)
                    cursor += n
                }
            }
        }
    }

    /** Renders the current mix to a 16-bit WAV in Music/VocalSplitter. Returns the display path. */
    fun export(
        ctx: Context,
        vocals: File,
        inst: File,
        totalFrames: Long,
        vocalGain: Float,
        instGain: Float,
        baseName: String,
        label: String,
    ): String {
        var peak = 0f
        forEachMixBlock(vocals, inst, totalFrames, vocalGain, instGain) { buf, count ->
            for (j in 0 until count) peak = max(peak, abs(buf[j]))
        }
        val scale = if (peak > 0.999f) 0.999f / peak else 1f

        val clean = baseName.replace(Regex("[\\\\/:*?\"<>|%\\r\\n]"), "_").trim().take(80).ifEmpty { "song" }
        val fileName = "${clean}_$label.wav"

        val resolver = ctx.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/x-wav")
            put(MediaStore.Audio.Media.RELATIVE_PATH, "Music/VocalSplitter")
            put(MediaStore.Audio.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Could not create the output file.")
        try {
            val out = resolver.openOutputStream(uri) ?: throw IOException("Could not open the output file.")
            out.buffered(1 shl 16).use { os ->
                val dataSize = (totalFrames * 4).toInt()
                val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                h.put("RIFF".toByteArray()); h.putInt(36 + dataSize)
                h.put("WAVE".toByteArray())
                h.put("fmt ".toByteArray()); h.putInt(16)
                h.putShort(1); h.putShort(2)
                h.putInt(SR); h.putInt(SR * 4)
                h.putShort(4); h.putShort(16)
                h.put("data".toByteArray()); h.putInt(dataSize)
                os.write(h.array())

                val bytes = ByteArray(BLOCK * 4)
                forEachMixBlock(vocals, inst, totalFrames, vocalGain, instGain) { buf, count ->
                    var p = 0
                    for (j in 0 until count) {
                        val s = (buf[j] * scale * 32767f).roundToInt().coerceIn(-32768, 32767)
                        bytes[p++] = (s and 0xFF).toByte()
                        bytes[p++] = ((s shr 8) and 0xFF).toByte()
                    }
                    os.write(bytes, 0, p)
                }
            }
            values.clear()
            values.put(MediaStore.Audio.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
        return "Music/VocalSplitter/$fileName"
    }
}
