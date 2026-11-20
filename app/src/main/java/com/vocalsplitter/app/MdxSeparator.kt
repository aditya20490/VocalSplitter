package com.vocalsplitter.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import kotlinx.coroutines.CancellationException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.cos

enum class Mode { BEST, FAST, SUPER }

/**
 * MDX-Net (UVR-MDX-NET-Voc_FT) vocal separation.
 * BEST: every chunk runs twice (normal + sign-flipped "denoise" pass) and is averaged.
 * FAST / SUPER: single pass. SUPER also uses all CPU cores.
 */
class MdxSeparator(modelPath: String, threads: Int) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val hop = 1024
    private val dimF: Int
    private val dimT: Int
    private val nFft: Int
    private val compensate: Float

    private val half: Int
    private val chunk: Int
    private val trim: Int
    private val gen: Int
    private val specSize: Int

    private val window: DoubleArray
    private val envelope: DoubleArray
    private val fft: Fft

    private val frameRe: DoubleArray
    private val frameIm: DoubleArray
    private val outRe: DoubleArray
    private val outIm: DoubleArray
    private val acc: DoubleArray
    private val inBuf: FloatBuffer

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads.coerceAtLeast(1))
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelPath, opts)
        inputName = session.inputNames.iterator().next()

        val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
        dimF = if (shape.size == 4 && shape[2] > 0) shape[2].toInt() else 3072
        dimT = if (shape.size == 4 && shape[3] > 0) shape[3].toInt() else 256
        if (dimF == 3072) {
            nFft = 7680
            compensate = 1.021f
        } else if (dimF == 2048) {
            nFft = 6144
            compensate = 1.035f
        } else {
            throw IllegalStateException("Unsupported model (frequency bins = $dimF)")
        }

        half = nFft / 2
        chunk = hop * (dimT - 1)
        trim = nFft / 2
        gen = chunk - 2 * trim
        specSize = 4 * dimF * dimT

        window = DoubleArray(nFft) { 0.5 - 0.5 * cos(2.0 * PI * it / nFft) }
        envelope = DoubleArray(chunk + nFft)
        for (t in 0 until dimT) {
            for (k in 0 until nFft) envelope[t * hop + k] += window[k] * window[k]
        }
        fft = Fft(nFft)
        frameRe = DoubleArray(nFft)
        frameIm = DoubleArray(nFft)
        outRe = DoubleArray(nFft)
        outIm = DoubleArray(nFft)
        acc = DoubleArray(chunk + nFft)
        inBuf = ByteBuffer.allocateDirect(specSize * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    private fun reflect(i: Int, n: Int): Int = if (i < 0) -i else if (i >= n) 2 * (n - 1) - i else i

    private fun computeSpec(l: FloatArray, r: FloatArray, spec: FloatArray) {
        for (c in 0..1) {
            val sig = if (c == 0) l else r
            for (t in 0 until dimT) {
                for (k in 0 until nFft) {
                    val pos = t * hop + k - half
                    frameRe[k] = sig[reflect(pos, chunk)] * window[k]
                    frameIm[k] = 0.0
                }
                fft.transform(frameRe, frameIm, outRe, outIm, false)
                for (f in 0 until dimF) {
                    spec[((c * 2) * dimF + f) * dimT + t] = outRe[f].toFloat()
                    spec[((c * 2 + 1) * dimF + f) * dimT + t] = outIm[f].toFloat()
                }
            }
        }
    }

    private fun inverse(pred: FloatArray, resL: FloatArray, resR: FloatArray) {
        for (c in 0..1) {
            java.util.Arrays.fill(acc, 0.0)
            for (t in 0 until dimT) {
                java.util.Arrays.fill(frameRe, 0.0)
                java.util.Arrays.fill(frameIm, 0.0)
                for (f in 0 until dimF) {
                    frameRe[f] = pred[((c * 2) * dimF + f) * dimT + t].toDouble()
                    frameIm[f] = pred[((c * 2 + 1) * dimF + f) * dimT + t].toDouble()
                }
                frameIm[0] = 0.0
                for (f in 1 until dimF) {
                    frameRe[nFft - f] = frameRe[f]
                    frameIm[nFft - f] = -frameIm[f]
                }
                fft.transform(frameRe, frameIm, outRe, outIm, true)
                val base = t * hop
                for (k in 0 until nFft) acc[base + k] += outRe[k] / nFft * window[k]
            }
            val dst = if (c == 0) resL else resR
            for (i in 0 until chunk) dst[i] = (acc[i + half] / envelope[i + half]).toFloat()
        }
    }

    private fun runModel(input: FloatArray, output: FloatArray) {
        inBuf.clear()
        inBuf.put(input)
        inBuf.rewind()
        val shape = longArrayOf(1, 4, dimF.toLong(), dimT.toLong())
        OnnxTensor.createTensor(env, inBuf, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { res ->
                val fb = (res[0] as OnnxTensor).floatBuffer
                fb.rewind()
                fb.get(output, 0, specSize)
            }
        }
    }

    /** Returns [vocalsLeft, vocalsRight]. */
    fun separate(
        left: FloatArray,
        right: FloatArray,
        denoise: Boolean,
        onProgress: (done: Int, total: Int) -> Unit,
        isActive: () -> Boolean,
    ): Array<FloatArray> {
        val n = left.size
        val total = (n + gen - 1) / gen
        val vocL = FloatArray(n)
        val vocR = FloatArray(n)

        val chL = FloatArray(chunk)
        val chR = FloatArray(chunk)
        val spec = FloatArray(specSize)
        val predA = FloatArray(specSize)
        val predB = if (denoise) FloatArray(specSize) else FloatArray(0)
        val resL = FloatArray(chunk)
        val resR = FloatArray(chunk)

        for (k in 0 until total) {
            if (!isActive()) throw CancellationException("cancelled")

            val start = k * gen - trim
            for (j in 0 until chunk) {
                val idx = start + j
                if (idx in 0 until n) {
                    chL[j] = left[idx]
                    chR[j] = right[idx]
                } else {
                    chL[j] = 0f
                    chR[j] = 0f
                }
            }

            computeSpec(chL, chR, spec)
            runModel(spec, predA)
            if (denoise) {
                for (i in 0 until specSize) spec[i] = -spec[i]
                runModel(spec, predB)
                for (i in 0 until specSize) predA[i] = 0.5f * (predA[i] - predB[i])
            }
            inverse(predA, resL, resR)

            val outStart = k * gen
            val count = minOf(gen, n - outStart)
            for (i in 0 until count) {
                vocL[outStart + i] = resL[trim + i] * compensate
                vocR[outStart + i] = resR[trim + i] * compensate
            }
            onProgress(k + 1, total)
        }
        return arrayOf(vocL, vocR)
    }

    override fun close() {
        try { session.close() } catch (_: Exception) {}
    }
}
