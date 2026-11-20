package com.vocalsplitter.app

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Mixed-radix complex FFT, works for any size (e.g. 7680 = 2^9 * 3 * 5). */
class Fft(private val n: Int) {
    private val factors: IntArray
    private val twRe = DoubleArray(n)
    private val twIm = DoubleArray(n)
    private val tmpRe: Array<DoubleArray>
    private val tmpIm: Array<DoubleArray>

    init {
        val list = ArrayList<Int>()
        var rest = n
        var p = 2
        while (rest > 1) {
            while (rest % p == 0) {
                list.add(p)
                rest /= p
            }
            p++
        }
        factors = list.toIntArray()
        for (k in 0 until n) {
            val a = 2.0 * PI * k / n
            twRe[k] = cos(a)
            twIm[k] = -sin(a)
        }
        tmpRe = Array(factors.size) { DoubleArray(factors[it]) }
        tmpIm = Array(factors.size) { DoubleArray(factors[it]) }
    }

    /** Unnormalised transform. inverse = true uses the conjugate twiddles (no 1/n scaling). */
    fun transform(
        inRe: DoubleArray, inIm: DoubleArray,
        outRe: DoubleArray, outIm: DoubleArray,
        inverse: Boolean,
    ) {
        rec(inRe, inIm, 0, 1, outRe, outIm, 0, n, 0, inverse)
    }

    private fun rec(
        inRe: DoubleArray, inIm: DoubleArray, inOff: Int, inStride: Int,
        outRe: DoubleArray, outIm: DoubleArray, outOff: Int,
        size: Int, fIdx: Int, inverse: Boolean,
    ) {
        if (size == 1) {
            outRe[outOff] = inRe[inOff]
            outIm[outOff] = inIm[inOff]
            return
        }
        val p = factors[fIdx]
        val m = size / p
        for (r in 0 until p) {
            rec(inRe, inIm, inOff + r * inStride, inStride * p, outRe, outIm, outOff + r * m, m, fIdx + 1, inverse)
        }
        val tr = tmpRe[fIdx]
        val ti = tmpIm[fIdx]
        val step = n / size
        for (k in 0 until m) {
            for (r in 0 until p) {
                val o = outOff + r * m + k
                tr[r] = outRe[o]
                ti[r] = outIm[o]
            }
            for (q in 0 until p) {
                val kk = k + q * m
                var sr = 0.0
                var si = 0.0
                for (r in 0 until p) {
                    val e = ((r * kk) % size) * step
                    val wr = twRe[e]
                    val wi = if (inverse) -twIm[e] else twIm[e]
                    sr += tr[r] * wr - ti[r] * wi
                    si += tr[r] * wi + ti[r] * wr
                }
                outRe[outOff + kk] = sr
                outIm[outOff + kk] = si
            }
        }
    }
}
