package com.vocalsplitter.app

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class Phase { Idle, Working, Done, Error }

data class StemResult(val name: String, val frames: Long)

data class UiState(
    val phase: Phase = Phase.Idle,
    val progress: Float = 0f,
    val message: String = "",
    val etaSec: Long = -1L,
    val result: StemResult? = null,
    val error: String? = null,
)

object SeparationState {
    val ui = MutableStateFlow(UiState())

    private fun dir(ctx: Context): File = File(ctx.filesDir, "stems").apply { mkdirs() }
    fun vocalsFile(ctx: Context) = File(dir(ctx), "vocals.f32")
    fun instFile(ctx: Context) = File(dir(ctx), "instrumental.f32")
    private fun metaFile(ctx: Context) = File(dir(ctx), "meta.txt")

    fun clear(ctx: Context) {
        vocalsFile(ctx).delete()
        instFile(ctx).delete()
        metaFile(ctx).delete()
    }

    fun saveMeta(ctx: Context, r: StemResult) {
        metaFile(ctx).writeText(r.name.replace("\n", " ") + "\n" + r.frames)
    }

    fun restore(ctx: Context) {
        if (ui.value.phase != Phase.Idle) return
        try {
            val lines = metaFile(ctx).readLines()
            val name = lines[0]
            val frames = lines[1].trim().toLong()
            if (vocalsFile(ctx).length() == frames * 8 && instFile(ctx).length() == frames * 8) {
                ui.value = UiState(
                    phase = Phase.Done,
                    progress = 1f,
                    message = "Ready",
                    result = StemResult(name, frames),
                )
            }
        } catch (_: Exception) {
        }
    }
}

/** Stems are stored as interleaved little-endian float32 stereo (lossless). */
object StemIO {
    fun write(file: File, a0: FloatArray, a1: FloatArray, s0: FloatArray?, s1: FloatArray?) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        val n = a0.size
        val block = 16384
        val bb = ByteBuffer.allocate(block * 8).order(ByteOrder.LITTLE_ENDIAN)
        FileOutputStream(tmp).channel.use { ch ->
            var pos = 0
            while (pos < n) {
                val cnt = minOf(block, n - pos)
                bb.clear()
                for (i in 0 until cnt) {
                    var l = a0[pos + i]
                    var r = a1[pos + i]
                    if (s0 != null && s1 != null) {
                        l -= s0[pos + i]
                        r -= s1[pos + i]
                    }
                    bb.putFloat(l)
                    bb.putFloat(r)
                }
                bb.flip()
                while (bb.hasRemaining()) ch.write(bb)
                pos += cnt
            }
        }
        if (file.exists()) file.delete()
        if (!tmp.renameTo(file)) throw IOException("Could not save " + file.name)
    }
}
