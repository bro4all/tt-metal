// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Helper function for _sfpu_binary_power_
// This function convert a float32 to int32, given that in >= 0.0f.
sfpi_inline sfpi::vInt _float_to_int32_positive_(sfpi::vFloat in) {
    sfpi::vInt result;
    sfpi::vInt exp = exexp(in);  // extract exponent
    v_if(exp < 0) { result = 0; }
    v_elseif(exp > 30)  // overflow occurs above this range
    {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<int32_t>::max();
    }
    v_else {
        // extract mantissa
        sfpi::vInt man = exman8(in);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23;  // 23 is number of mantissa in float32
        man = shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);

        result = man;
    }
    v_endif;
    return result;
}

// Helper function for _sfpu_binary_power_
// This function convert a float32 to int32, given that in >= 0.0f.
// sfpi_inline sfpi::vInt _float_to_int32_positive_alt_(sfpi::vFloat in) {
//     sfpi::vInt result = 0;
//     sfpi::vInt exp = 0; // exexp(in);  // extract exponent

//     // extract mantissa
//     // sfpi::vInt man = exman8(in);
//     // // shift the mantissa by (23-exponent) to the right
//     // sfpi::vInt shift = exp - 23;  // 23 is number of mantissa in float32
//     // man = shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);

//     // result = man;

//     v_if(exp < 0) { result = 0; }
//     v_endif;

//     // v_if(exp > 30)  // overflow occurs above this range
//     // {
//     //     // set to int32 max value in case of overflow
//     //     result = std::numeric_limits<int32_t>::max();
//     // }
//     // v_endif;

//     return result;
// }

// sfpi_inline sfpi::vFloat _sfpu_binary_power_alt0_(sfpi::vFloat base, sfpi::vFloat pow) {

//     // POW = 2.0, karma should be 2.f as well
//     sfpi::vFloat karma = pow;
//     v_if (pow < 2.f) {
//         karma = 100.f;
//     }
//     v_endif;

//     sfpi::vFloat result = sfpi::int32_to_float(_float_to_int32_positive_alt_(100.f));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     result = sfpi::int32_to_float(_float_to_int32_positive_alt_(result));
//     return karma;

//     // FAIL, returns 100.f
// }

// sfpi_inline sfpi::vFloat conditional(sfpi::vFloat in) {
//     sfpi::vFloat result = in;
//     // sfpi::vInt exp = exexp(in); // exexp(in);  // extract exponent

//     // extract mantissa
//     // sfpi::vInt man = exman8(in);
//     // // shift the mantissa by (23-exponent) to the right
//     // sfpi::vInt shift = exp - 23;  // 23 is number of mantissa in float32
//     // man = shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);

//     // result = man;

//     v_if (in < 0.f) {
//         result = 0.f;
//     }
//     v_endif;

//     // v_if(exp < 0) { result = 0.f; }
//     // v_endif;

//     // v_if(exp > 30)  // overflow occurs above this range
//     // {
//     //     // set to int32 max value in case of overflow
//     //     result = std::numeric_limits<int32_t>::max();
//     // }
//     // v_endif;

//     return result;
// }

// sfpi_inline sfpi::vFloat test_conditional_bug0(sfpi::vFloat base, sfpi::vFloat pow) {

//     // POW = 2.0, karma should be 2.f as well
//     sfpi::vFloat karma = pow;
//     v_if (pow < 2.f) {
//         karma = 100.f;
//     }
//     v_endif;

//     sfpi::vFloat result;
//     result = conditional(100.f);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);
//     result = conditional(result);

//     return karma;

//     // FAIL, returns 100.f
// }

sfpi_inline sfpi::vFloat process(sfpi::vFloat in) { return in + 3.1415; }

sfpi_inline sfpi::vFloat test_conditional_bug0(sfpi::vFloat base, sfpi::vFloat pow) {
    // POW = 2.0, karma should be 2.f as well
    sfpi::vFloat independent = pow;
    v_if(pow < 2.f) { independent = 100.f; }
    v_endif;

    sfpi::vFloat result = 100.f;
    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    v_if(result < 0.f) { result = process(result); }
    v_endif;

    // v_if (result < 0.f) {
    //     result = process(result);
    // }
    // v_endif;

    // v_if (result < 0.f) {
    //     result = process(result);
    // }
    // v_endif;

    return independent;
}

// POW = 2.0, karma should be 2.f as well
sfpi_inline sfpi::vFloat test_conditional_bug(sfpi::vFloat base, sfpi::vFloat pow) {
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();

    return pow;
}

sfpi_inline sfpi::vFloat process_alt0(sfpi::vFloat in) { return 0.5f * (3.0f * (in * 0.333333f) + 5.f * (in * 0.2f)); }

sfpi_inline sfpi::vFloat test_conditional_bug2(sfpi::vFloat base, sfpi::vFloat pow) {
    // POW = 2.0, karma should be 2.f as well
    sfpi::vInt independent = reinterpret<sfpi::vInt>(pow);
    v_if(pow < 2.f) { independent = sfpi::vInt(0xDEADBEEF); }
    v_endif;

    sfpi::vFloat result = 100.f;
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);
    result = process_alt0(100.f);

    return reinterpret<sfpi::vFloat>(independent);
}

sfpi_inline sfpi::vFloat test_conditional_bug3(sfpi::vFloat base, sfpi::vFloat pow) {
    // POW = 2.0, karma should be 2.f as well
    sfpi::vFloat independent = pow;
    v_if(pow < 2.f) { independent = 200.f; }
    v_endif;

    sfpi::vFloat result = 100.f;
    result = sfpi::int32_to_float(_float_to_int32_positive_(result));
    result = sfpi::int32_to_float(_float_to_int32_positive_(result));
    result = sfpi::int32_to_float(_float_to_int32_positive_(result));

    return independent;
}

// sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow) {
//     // Normalize base to calculation range
//     sfpi::vFloat x = setsgn(base, 0);  // set base as positive
//     x = sfpi::setexp(x, 127);          // set exp to exp bias (put base in range of 1-2)

//     // 3rd order polynomial approx - determined using rminimax over [1,2]
//     sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

//     // Convert exponent to float
//     sfpi::vInt exp = sfpi::exexp(base);
//     v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
//     v_endif;
//     sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

//     // De-normalize to original range
//     const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;        // 1.4426950408889634f;
//     sfpi::vFloat log2_result = expf + series_result * vConst1Ln2;  // exp correction: ln(1+x) + exp*ln(2)

//     sfpi::vFloat zff = pow * log2_result;
//     const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;
//     v_if(zff < low_threshold)  // -126.99999237060546875
//     {
//         zff = low_threshold;
//     }
//     v_endif;

//     zff = addexp(zff, 23);                                                 // * 2**23 (Mn)
//     sfpi::vInt z = _float_to_int32_positive_(zff + sfpi::vFloat(0x3f800000));  // (bias + x * log2(a)) * N_m

//     sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));
//     sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

//     sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
//     sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
//     sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
//     d2 = d1 * d2;
//     zif = _float_to_int32_positive_(d2 * d3);

//     // restore exponent
//     zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

//     sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

//     // Check valid base range
//     sfpi::vInt pow_int =
//         sfpi::float_to_int16(pow, 0);  // int16 should be plenty, since large powers will approach 0/Inf
//     sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

//     v_if(base == 0.f && pow < 0.f) {
//         y = std::numeric_limits<float>::quiet_NaN();  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
//     }
//     v_endif

//     v_if(base < 0.0f) {  // negative base
//         // Check for integer power
//         v_if(pow_rounded == pow) {
//             // if pow is odd integer, set result to negative
//             v_if(pow_int & 0x1) {
//                 // if negative base and negative pow then x**y = -(abs(x))**(abs(y))
//                 // `sign` will be used at the end
//                 y = setsgn(y, 1);
//             }
//             v_endif;
//         }
//         v_else {
//             // multiplication by NaN gives NaN.
//             // Since we are going to multiply the result by `sign` to handle negative bases, we also use
//             // `sign` to handle NaN results
//             y = std::numeric_limits<float>::quiet_NaN();
//         }
//         v_endif;
//     }
//     v_endif;

//     return y;
// }  // namespace ckernel

// sfpi_inline sfpi::vFloat _sfpu_binary_power_alt2_(sfpi::vFloat base, sfpi::vFloat pow) {
//     // Normalize base to calculation range

//     // sfpi::vFloat x = sfpi::setexp(x, 127);          // set exp to exp bias (put base in range of 1-2)

//     // 3rd order polynomial approx - determined using rminimax over [1,2]
//     // sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

//     // // Convert exponent to float
//     // sfpi::vInt exp = sfpi::exexp(base);
//     // // v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
//     // // v_endif;
//     // sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

//     // // De-normalize to original range
//     // // const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;        // 1.4426950408889634f;
//     // sfpi::vFloat log2_result = expf + series_result * 1.442695f;  // exp correction: ln(1+x) + exp*ln(2)

//     sfpi::vFloat karma = 2.f;
//     // pow = 2.f;
//     v_if (pow < 2.f) {
//         karma = 20.f;
//     }
//     v_endif;

//     // return karma;

//     // pow = 2.f;
//     // pow = sfpi::vConst1 + sfpi::vConst1;
//     // return pow;
//     sfpi::vFloat log2_result = 3.169925001442312; // OK
//     sfpi::vFloat zff = karma * log2_result; // OK

//     // const sfpi::vFloat low_threshold = -127.0f;
//     // v_if(zff < low_threshold)  // -126.99999237060546875
//     // {
//     //     zff = low_threshold;
//     // }
//     // v_endif;

//     zff = addexp(zff, 23);                                                 // * 2**23 (Mn) // OK

//     zff =  zff + 1065353216.f; // OK
//     // zff = 1118535313.f;

//     // sfpi::vInt z = _float_to_int32_positive_(zff);  // (bias + x * log2(a)) * N_m // OK

//     sfpi::vInt z = 0x42AB7E90;

//     // sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z)); // OK
//     sfpi::vInt zii = z & 0x7f800000;
//     sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z)); // OK (?)
//     // sfpi::vInt zif = sfpi::exman8(sfpi::reinterpret<sfpi::vFloat>(z)) & 0x7fffff;

//     sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7); // OK
//     sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif); // OK

//     sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif); // OK
//     d2 = d1 * d2; // OK

//     // return karma; // return 2.f

//     // zif = _float_to_int32_positive_(d2 * d3); // OK
//     // return sfpi::int32_to_float(zif);

//     zif = foo(100.f);
//     // zif = foo(100.f);
//     // zif = _float_to_int32_positive_alt_(100.f); // OK
//     // zif = _float_to_int32_positive_alt_(100.f); // OK

//     return karma; // return 20.f;

//     // return sfpi::setexp(sfpi::vFloat(1.5f), 127U + zii);

//     // zif = sfpi::reinterpret<sfpi::vInt>(sfpi::vFloat(1.5f));

//     // restore exponent
//     // return sfpi::reinterpret<sfpi::vFloat>(zii);

//     // return sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vUInt>(zii) |
//     sfpi::reinterpret<sfpi::vUInt>(zif));

//     // Result should be 0x42a191d1 ~= 80.78..

//     // return sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(0x42800000) | sfpi::reinterpret<sfpi::vUInt>(zif));
//     // return sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(0x42800000) | sfpi::vUInt(0x002191d1));

//     zii |= zif;

//     // zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127 +
//     sfpi::reinterpret<sfpi::vInt>(zii)));

//     // return sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(0x42800000) | sfpi::vUInt(0x002191d1));

//     sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

//     // Check valid base range
//     // sfpi::vInt pow_int =
//     //     sfpi::flßßoat_to_int16(pow, 0);  // int16 should be plenty, since large powers will approach 0/Inf
//     // sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

//     // v_if(base == 0.f && pow < 0.f) {
//     //     y = std::numeric_limits<float>::quiet_NaN();  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
//     // }
//     // v_endif

//     // v_if(base < 0.0f) {  // negative base
//     //     // Check for integer power
//     //     v_if(pow_rounded == pow) {
//     //         // if pow is odd integer, set result to negative
//     //         v_if(pow_int & 0x1) {
//     //             // if negative base and negative pow then x**y = -(abs(x))**(abs(y))
//     //             // `sign` will be used at the end
//     //             y = setsgn(y, 1);
//     //         }
//     //         v_endif;
//     //     }
//     //     v_else {
//     //         // multiplication by NaN gives NaN.
//     //         // Since we are going to multiply the result by `sign` to handle negative bases, we also use
//     //         // `sign` to handle NaN results
//     //         y = std::numeric_limits<float>::quiet_NaN();
//     //     }
//     //     v_endif;
//     // }
//     // v_endif;

//     // return 1.55f;
//     return y;
// }  // namespace ckernel

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void calculate_sfpu_binary(const uint dst_offset) {
    if constexpr (BINOP == BinaryOp::POW) {
        TTI_SFPCONFIG(0, 11, 1);
        for (int d = 0; d < ITERATIONS; d++) {
            constexpr uint dst_tile_size = 32;
            sfpi::vFloat in0 = sfpi::dst_reg[0];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_offset * dst_tile_size];

            sfpi::dst_reg[0] = 0.f;  // DEBUG-only

            sfpi::vFloat result = test_conditional_bug(in0, in1);

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_offset);
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::POW) {
        TTI_SFPCONFIG(0, 11, 1);
        sfpi::vConstFloatPrgm0 = 2.f;
        sfpi::vConstFloatPrgm1 = 1.442695f;
        // sfpi::vConstFloatPrgm2 = -127.0f;

    } else {
        _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
