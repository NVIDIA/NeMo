#ifndef CUFFTDX_FFT_15625_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_15625_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1185, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.local .align 8 .b8 __local_depot0[200];
.reg .b64 SP;
.reg .b64 SPL;
.reg .pred p<3>;
.reg .f32 f<695>;
.reg .b32 r<11414>;
.reg .b64 rd<21>;
mov.u64 SPL, __local_depot0;
add.u64 rd3, SPL, 0;
mov.u32 r3551, %tid.y;
mul.lo.s32 r1, r3551, 15625;
add.s64 rd4, rd3, 4;
mov.f32 f214, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r23, {low, high};
}
mov.f32 f216, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r24, {low, high};
}
{
neg.f16x2 r25, r24;
}
mov.f32 f210, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r27, {low, high};
}
mov.f32 f212, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r28, {low, high};
}
{
neg.f16x2 r29, r28;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r31, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r32, {low, high};
}
{
add.f16x2 r33, %61, %91;
}
{
add.f16x2 r36, %51, r33;
}
{
add.f16x2 r39, %71, %81;
}
{
add.f16x2 r42, r36, r39;
}
{
add.f16x2 r45, %62, %92;
}
{
add.f16x2 r48, %52, r45;
}
{
add.f16x2 r51, %72, %82;
}
{
add.f16x2 r54, r48, r51;
}
{
add.f16x2 r57, %61, %91;
}
{
mul.f16x2 r60, r57, r23;
}
{
add.f16x2 r63, %51, r60;
}
{
add.f16x2 r66, %71, %81;
}
{
mul.f16x2 r69, r66, r27;
}
{
add.f16x2 r72, r63, r69;
}
{
sub.f16x2 r75, %62, %92;
}
{
mul.f16x2 r78, r75, r25;
}
{
sub.f16x2 r81, %72, %82;
}
{
mul.f16x2 r84, r81, r29;
}
{
add.f16x2 r87, r78, r84;
}
{
sub.f16x2 r90, r72, r87;
}
{
add.f16x2 r93, %61, %91;
}
{
mul.f16x2 r96, r93, r23;
}
{
add.f16x2 r99, %51, r96;
}
{
add.f16x2 r102, %71, %81;
}
{
mul.f16x2 r105, r102, r27;
}
{
add.f16x2 r108, r99, r105;
}
{
sub.f16x2 r111, %62, %92;
}
{
mul.f16x2 r114, r111, r25;
}
{
sub.f16x2 r117, %72, %82;
}
{
mul.f16x2 r120, r117, r29;
}
{
add.f16x2 r123, r114, r120;
}
{
add.f16x2 r126, r108, r123;
}
{
add.f16x2 r129, %61, %91;
}
{
mul.f16x2 r132, r129, r27;
}
{
add.f16x2 r135, %51, r132;
}
{
add.f16x2 r138, %71, %81;
}
{
mul.f16x2 r141, r138, r31;
}
{
add.f16x2 r144, r135, r141;
}
{
sub.f16x2 r147, %62, %92;
}
{
mul.f16x2 r150, r147, r29;
}
{
sub.f16x2 r153, %72, %82;
}
{
mul.f16x2 r156, r153, r32;
}
{
add.f16x2 r159, r150, r156;
}
{
sub.f16x2 r162, r144, r159;
}
{
add.f16x2 r165, %61, %91;
}
{
mul.f16x2 r168, r165, r27;
}
{
add.f16x2 r171, %51, r168;
}
{
add.f16x2 r174, %71, %81;
}
{
mul.f16x2 r177, r174, r31;
}
{
add.f16x2 r180, r171, r177;
}
{
sub.f16x2 r183, %62, %92;
}
{
mul.f16x2 r186, r183, r29;
}
{
sub.f16x2 r189, %72, %82;
}
{
mul.f16x2 r192, r189, r32;
}
{
add.f16x2 r195, r186, r192;
}
{
add.f16x2 r198, r180, r195;
}
{
add.f16x2 r201, %62, %92;
}
{
mul.f16x2 r204, r201, r23;
}
{
add.f16x2 r207, %52, r204;
}
{
add.f16x2 r210, %72, %82;
}
{
mul.f16x2 r213, r210, r27;
}
{
add.f16x2 r216, r207, r213;
}
{
sub.f16x2 r219, %61, %91;
}
{
mul.f16x2 r222, r219, r25;
}
{
sub.f16x2 r225, %71, %81;
}
{
mul.f16x2 r228, r225, r29;
}
{
add.f16x2 r231, r222, r228;
}
{
add.f16x2 r234, r216, r231;
}
{
add.f16x2 r237, %62, %92;
}
{
mul.f16x2 r240, r237, r23;
}
{
add.f16x2 r243, %52, r240;
}
{
add.f16x2 r246, %72, %82;
}
{
mul.f16x2 r249, r246, r27;
}
{
add.f16x2 r252, r243, r249;
}
{
sub.f16x2 r255, %61, %91;
}
{
mul.f16x2 r258, r255, r25;
}
{
sub.f16x2 r261, %71, %81;
}
{
mul.f16x2 r264, r261, r29;
}
{
add.f16x2 r267, r258, r264;
}
{
sub.f16x2 r270, r252, r267;
}
{
add.f16x2 r273, %62, %92;
}
{
mul.f16x2 r276, r273, r27;
}
{
add.f16x2 r279, %52, r276;
}
{
add.f16x2 r282, %72, %82;
}
{
mul.f16x2 r285, r282, r31;
}
{
add.f16x2 r288, r279, r285;
}
{
sub.f16x2 r291, %61, %91;
}
{
mul.f16x2 r294, r291, r29;
}
{
sub.f16x2 r297, %71, %81;
}
{
mul.f16x2 r300, r297, r32;
}
{
add.f16x2 r303, r294, r300;
}
{
add.f16x2 r306, r288, r303;
}
{
add.f16x2 r309, %62, %92;
}
{
mul.f16x2 r312, r309, r27;
}
{
add.f16x2 r315, %52, r312;
}
{
add.f16x2 r318, %72, %82;
}
{
mul.f16x2 r321, r318, r31;
}
{
add.f16x2 r324, r315, r321;
}
{
sub.f16x2 r327, %61, %91;
}
{
mul.f16x2 r330, r327, r29;
}
{
sub.f16x2 r333, %71, %81;
}
{
mul.f16x2 r336, r333, r32;
}
{
add.f16x2 r339, r330, r336;
}
{
sub.f16x2 r342, r324, r339;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r349, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r350, {low, high};
}
{
neg.f16x2 r351, r350;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r353, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r354, {low, high};
}
{
add.f16x2 r355, %63, %93;
}
{
add.f16x2 r358, %53, r355;
}
{
add.f16x2 r361, %73, %83;
}
{
add.f16x2 r364, r358, r361;
}
{
add.f16x2 r367, %64, %94;
}
{
add.f16x2 r370, %54, r367;
}
{
add.f16x2 r373, %74, %84;
}
{
add.f16x2 r376, r370, r373;
}
{
add.f16x2 r379, %63, %93;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, %53, r382;
}
{
add.f16x2 r388, %73, %83;
}
{
mul.f16x2 r391, r388, r349;
}
{
add.f16x2 r394, r385, r391;
}
{
sub.f16x2 r397, %64, %94;
}
{
mul.f16x2 r400, r397, r347;
}
{
sub.f16x2 r403, %74, %84;
}
{
mul.f16x2 r406, r403, r351;
}
{
add.f16x2 r409, r400, r406;
}
{
sub.f16x2 r412, r394, r409;
}
{
add.f16x2 r415, %63, %93;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, %53, r418;
}
{
add.f16x2 r424, %73, %83;
}
{
mul.f16x2 r427, r424, r349;
}
{
add.f16x2 r430, r421, r427;
}
{
sub.f16x2 r433, %64, %94;
}
{
mul.f16x2 r436, r433, r347;
}
{
sub.f16x2 r439, %74, %84;
}
{
mul.f16x2 r442, r439, r351;
}
{
add.f16x2 r445, r436, r442;
}
{
add.f16x2 r448, r430, r445;
}
{
add.f16x2 r451, %63, %93;
}
{
mul.f16x2 r454, r451, r349;
}
{
add.f16x2 r457, %53, r454;
}
{
add.f16x2 r460, %73, %83;
}
{
mul.f16x2 r463, r460, r353;
}
{
add.f16x2 r466, r457, r463;
}
{
sub.f16x2 r469, %64, %94;
}
{
mul.f16x2 r472, r469, r351;
}
{
sub.f16x2 r475, %74, %84;
}
{
mul.f16x2 r478, r475, r354;
}
{
add.f16x2 r481, r472, r478;
}
{
sub.f16x2 r484, r466, r481;
}
{
add.f16x2 r487, %63, %93;
}
{
mul.f16x2 r490, r487, r349;
}
{
add.f16x2 r493, %53, r490;
}
{
add.f16x2 r496, %73, %83;
}
{
mul.f16x2 r499, r496, r353;
}
{
add.f16x2 r502, r493, r499;
}
{
sub.f16x2 r505, %64, %94;
}
{
mul.f16x2 r508, r505, r351;
}
{
sub.f16x2 r511, %74, %84;
}
{
mul.f16x2 r514, r511, r354;
}
{
add.f16x2 r517, r508, r514;
}
{
add.f16x2 r520, r502, r517;
}
{
add.f16x2 r523, %64, %94;
}
{
mul.f16x2 r526, r523, r345;
}
{
add.f16x2 r529, %54, r526;
}
{
add.f16x2 r532, %74, %84;
}
{
mul.f16x2 r535, r532, r349;
}
{
add.f16x2 r538, r529, r535;
}
{
sub.f16x2 r541, %63, %93;
}
{
mul.f16x2 r544, r541, r347;
}
{
sub.f16x2 r547, %73, %83;
}
{
mul.f16x2 r550, r547, r351;
}
{
add.f16x2 r553, r544, r550;
}
{
add.f16x2 r556, r538, r553;
}
{
add.f16x2 r559, %64, %94;
}
{
mul.f16x2 r562, r559, r345;
}
{
add.f16x2 r565, %54, r562;
}
{
add.f16x2 r568, %74, %84;
}
{
mul.f16x2 r571, r568, r349;
}
{
add.f16x2 r574, r565, r571;
}
{
sub.f16x2 r577, %63, %93;
}
{
mul.f16x2 r580, r577, r347;
}
{
sub.f16x2 r583, %73, %83;
}
{
mul.f16x2 r586, r583, r351;
}
{
add.f16x2 r589, r580, r586;
}
{
sub.f16x2 r592, r574, r589;
}
{
add.f16x2 r595, %64, %94;
}
{
mul.f16x2 r598, r595, r349;
}
{
add.f16x2 r601, %54, r598;
}
{
add.f16x2 r604, %74, %84;
}
{
mul.f16x2 r607, r604, r353;
}
{
add.f16x2 r610, r601, r607;
}
{
sub.f16x2 r613, %63, %93;
}
{
mul.f16x2 r616, r613, r351;
}
{
sub.f16x2 r619, %73, %83;
}
{
mul.f16x2 r622, r619, r354;
}
{
add.f16x2 r625, r616, r622;
}
{
add.f16x2 r628, r610, r625;
}
{
add.f16x2 r631, %64, %94;
}
{
mul.f16x2 r634, r631, r349;
}
{
add.f16x2 r637, %54, r634;
}
{
add.f16x2 r640, %74, %84;
}
{
mul.f16x2 r643, r640, r353;
}
{
add.f16x2 r646, r637, r643;
}
{
sub.f16x2 r649, %63, %93;
}
{
mul.f16x2 r652, r649, r351;
}
{
sub.f16x2 r655, %73, %83;
}
{
mul.f16x2 r658, r655, r354;
}
{
add.f16x2 r661, r652, r658;
}
{
sub.f16x2 r664, r646, r661;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r668, {low, high};
}
{
neg.f16x2 r669, r668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r672, {low, high};
}
{
neg.f16x2 r673, r672;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r675, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r676, {low, high};
}
{
add.f16x2 r677, %65, %95;
}
{
add.f16x2 r680, %55, r677;
}
{
add.f16x2 r683, %75, %85;
}
{
add.f16x2 r686, r680, r683;
}
{
add.f16x2 r689, %66, %96;
}
{
add.f16x2 r692, %56, r689;
}
{
add.f16x2 r695, %76, %86;
}
{
add.f16x2 r698, r692, r695;
}
{
add.f16x2 r701, %65, %95;
}
{
mul.f16x2 r704, r701, r667;
}
{
add.f16x2 r707, %55, r704;
}
{
add.f16x2 r710, %75, %85;
}
{
mul.f16x2 r713, r710, r671;
}
{
add.f16x2 r716, r707, r713;
}
{
sub.f16x2 r719, %66, %96;
}
{
mul.f16x2 r722, r719, r669;
}
{
sub.f16x2 r725, %76, %86;
}
{
mul.f16x2 r728, r725, r673;
}
{
add.f16x2 r731, r722, r728;
}
{
sub.f16x2 r734, r716, r731;
}
{
add.f16x2 r737, %65, %95;
}
{
mul.f16x2 r740, r737, r667;
}
{
add.f16x2 r743, %55, r740;
}
{
add.f16x2 r746, %75, %85;
}
{
mul.f16x2 r749, r746, r671;
}
{
add.f16x2 r752, r743, r749;
}
{
sub.f16x2 r755, %66, %96;
}
{
mul.f16x2 r758, r755, r669;
}
{
sub.f16x2 r761, %76, %86;
}
{
mul.f16x2 r764, r761, r673;
}
{
add.f16x2 r767, r758, r764;
}
{
add.f16x2 r770, r752, r767;
}
{
add.f16x2 r773, %65, %95;
}
{
mul.f16x2 r776, r773, r671;
}
{
add.f16x2 r779, %55, r776;
}
{
add.f16x2 r782, %75, %85;
}
{
mul.f16x2 r785, r782, r675;
}
{
add.f16x2 r788, r779, r785;
}
{
sub.f16x2 r791, %66, %96;
}
{
mul.f16x2 r794, r791, r673;
}
{
sub.f16x2 r797, %76, %86;
}
{
mul.f16x2 r800, r797, r676;
}
{
add.f16x2 r803, r794, r800;
}
{
sub.f16x2 r806, r788, r803;
}
{
add.f16x2 r809, %65, %95;
}
{
mul.f16x2 r812, r809, r671;
}
{
add.f16x2 r815, %55, r812;
}
{
add.f16x2 r818, %75, %85;
}
{
mul.f16x2 r821, r818, r675;
}
{
add.f16x2 r824, r815, r821;
}
{
sub.f16x2 r827, %66, %96;
}
{
mul.f16x2 r830, r827, r673;
}
{
sub.f16x2 r833, %76, %86;
}
{
mul.f16x2 r836, r833, r676;
}
{
add.f16x2 r839, r830, r836;
}
{
add.f16x2 r842, r824, r839;
}
{
add.f16x2 r845, %66, %96;
}
{
mul.f16x2 r848, r845, r667;
}
{
add.f16x2 r851, %56, r848;
}
{
add.f16x2 r854, %76, %86;
}
{
mul.f16x2 r857, r854, r671;
}
{
add.f16x2 r860, r851, r857;
}
{
sub.f16x2 r863, %65, %95;
}
{
mul.f16x2 r866, r863, r669;
}
{
sub.f16x2 r869, %75, %85;
}
{
mul.f16x2 r872, r869, r673;
}
{
add.f16x2 r875, r866, r872;
}
{
add.f16x2 r878, r860, r875;
}
{
add.f16x2 r881, %66, %96;
}
{
mul.f16x2 r884, r881, r667;
}
{
add.f16x2 r887, %56, r884;
}
{
add.f16x2 r890, %76, %86;
}
{
mul.f16x2 r893, r890, r671;
}
{
add.f16x2 r896, r887, r893;
}
{
sub.f16x2 r899, %65, %95;
}
{
mul.f16x2 r902, r899, r669;
}
{
sub.f16x2 r905, %75, %85;
}
{
mul.f16x2 r908, r905, r673;
}
{
add.f16x2 r911, r902, r908;
}
{
sub.f16x2 r914, r896, r911;
}
{
add.f16x2 r917, %66, %96;
}
{
mul.f16x2 r920, r917, r671;
}
{
add.f16x2 r923, %56, r920;
}
{
add.f16x2 r926, %76, %86;
}
{
mul.f16x2 r929, r926, r675;
}
{
add.f16x2 r932, r923, r929;
}
{
sub.f16x2 r935, %65, %95;
}
{
mul.f16x2 r938, r935, r673;
}
{
sub.f16x2 r941, %75, %85;
}
{
mul.f16x2 r944, r941, r676;
}
{
add.f16x2 r947, r938, r944;
}
{
add.f16x2 r950, r932, r947;
}
{
add.f16x2 r953, %66, %96;
}
{
mul.f16x2 r956, r953, r671;
}
{
add.f16x2 r959, %56, r956;
}
{
add.f16x2 r962, %76, %86;
}
{
mul.f16x2 r965, r962, r675;
}
{
add.f16x2 r968, r959, r965;
}
{
sub.f16x2 r971, %65, %95;
}
{
mul.f16x2 r974, r971, r673;
}
{
sub.f16x2 r977, %75, %85;
}
{
mul.f16x2 r980, r977, r676;
}
{
add.f16x2 r983, r974, r980;
}
{
sub.f16x2 r986, r968, r983;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r989, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r990, {low, high};
}
{
neg.f16x2 r991, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r993, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r994, {low, high};
}
{
neg.f16x2 r995, r994;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r998, {low, high};
}
{
add.f16x2 r999, %67, %97;
}
{
add.f16x2 r1002, %57, r999;
}
{
add.f16x2 r1005, %77, %87;
}
{
add.f16x2 r1008, r1002, r1005;
}
{
add.f16x2 r1011, %68, %98;
}
{
add.f16x2 r1014, %58, r1011;
}
{
add.f16x2 r1017, %78, %88;
}
{
add.f16x2 r1020, r1014, r1017;
}
{
add.f16x2 r1023, %67, %97;
}
{
mul.f16x2 r1026, r1023, r989;
}
{
add.f16x2 r1029, %57, r1026;
}
{
add.f16x2 r1032, %77, %87;
}
{
mul.f16x2 r1035, r1032, r993;
}
{
add.f16x2 r1038, r1029, r1035;
}
{
sub.f16x2 r1041, %68, %98;
}
{
mul.f16x2 r1044, r1041, r991;
}
{
sub.f16x2 r1047, %78, %88;
}
{
mul.f16x2 r1050, r1047, r995;
}
{
add.f16x2 r1053, r1044, r1050;
}
{
sub.f16x2 r1056, r1038, r1053;
}
{
add.f16x2 r1059, %67, %97;
}
{
mul.f16x2 r1062, r1059, r989;
}
{
add.f16x2 r1065, %57, r1062;
}
{
add.f16x2 r1068, %77, %87;
}
{
mul.f16x2 r1071, r1068, r993;
}
{
add.f16x2 r1074, r1065, r1071;
}
{
sub.f16x2 r1077, %68, %98;
}
{
mul.f16x2 r1080, r1077, r991;
}
{
sub.f16x2 r1083, %78, %88;
}
{
mul.f16x2 r1086, r1083, r995;
}
{
add.f16x2 r1089, r1080, r1086;
}
{
add.f16x2 r1092, r1074, r1089;
}
{
add.f16x2 r1095, %67, %97;
}
{
mul.f16x2 r1098, r1095, r993;
}
{
add.f16x2 r1101, %57, r1098;
}
{
add.f16x2 r1104, %77, %87;
}
{
mul.f16x2 r1107, r1104, r997;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
sub.f16x2 r1113, %68, %98;
}
{
mul.f16x2 r1116, r1113, r995;
}
{
sub.f16x2 r1119, %78, %88;
}
{
mul.f16x2 r1122, r1119, r998;
}
{
add.f16x2 r1125, r1116, r1122;
}
{
sub.f16x2 r1128, r1110, r1125;
}
{
add.f16x2 r1131, %67, %97;
}
{
mul.f16x2 r1134, r1131, r993;
}
{
add.f16x2 r1137, %57, r1134;
}
{
add.f16x2 r1140, %77, %87;
}
{
mul.f16x2 r1143, r1140, r997;
}
{
add.f16x2 r1146, r1137, r1143;
}
{
sub.f16x2 r1149, %68, %98;
}
{
mul.f16x2 r1152, r1149, r995;
}
{
sub.f16x2 r1155, %78, %88;
}
{
mul.f16x2 r1158, r1155, r998;
}
{
add.f16x2 r1161, r1152, r1158;
}
{
add.f16x2 r1164, r1146, r1161;
}
{
add.f16x2 r1167, %68, %98;
}
{
mul.f16x2 r1170, r1167, r989;
}
{
add.f16x2 r1173, %58, r1170;
}
{
add.f16x2 r1176, %78, %88;
}
{
mul.f16x2 r1179, r1176, r993;
}
{
add.f16x2 r1182, r1173, r1179;
}
{
sub.f16x2 r1185, %67, %97;
}
{
mul.f16x2 r1188, r1185, r991;
}
{
sub.f16x2 r1191, %77, %87;
}
{
mul.f16x2 r1194, r1191, r995;
}
{
add.f16x2 r1197, r1188, r1194;
}
{
add.f16x2 r1200, r1182, r1197;
}
{
add.f16x2 r1203, %68, %98;
}
{
mul.f16x2 r1206, r1203, r989;
}
{
add.f16x2 r1209, %58, r1206;
}
{
add.f16x2 r1212, %78, %88;
}
{
mul.f16x2 r1215, r1212, r993;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
sub.f16x2 r1221, %67, %97;
}
{
mul.f16x2 r1224, r1221, r991;
}
{
sub.f16x2 r1227, %77, %87;
}
{
mul.f16x2 r1230, r1227, r995;
}
{
add.f16x2 r1233, r1224, r1230;
}
{
sub.f16x2 r1236, r1218, r1233;
}
{
add.f16x2 r1239, %68, %98;
}
{
mul.f16x2 r1242, r1239, r993;
}
{
add.f16x2 r1245, %58, r1242;
}
{
add.f16x2 r1248, %78, %88;
}
{
mul.f16x2 r1251, r1248, r997;
}
{
add.f16x2 r1254, r1245, r1251;
}
{
sub.f16x2 r1257, %67, %97;
}
{
mul.f16x2 r1260, r1257, r995;
}
{
sub.f16x2 r1263, %77, %87;
}
{
mul.f16x2 r1266, r1263, r998;
}
{
add.f16x2 r1269, r1260, r1266;
}
{
add.f16x2 r1272, r1254, r1269;
}
{
add.f16x2 r1275, %68, %98;
}
{
mul.f16x2 r1278, r1275, r993;
}
{
add.f16x2 r1281, %58, r1278;
}
{
add.f16x2 r1284, %78, %88;
}
{
mul.f16x2 r1287, r1284, r997;
}
{
add.f16x2 r1290, r1281, r1287;
}
{
sub.f16x2 r1293, %67, %97;
}
{
mul.f16x2 r1296, r1293, r995;
}
{
sub.f16x2 r1299, %77, %87;
}
{
mul.f16x2 r1302, r1299, r998;
}
{
add.f16x2 r1305, r1296, r1302;
}
{
sub.f16x2 r1308, r1290, r1305;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1311, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1312, {low, high};
}
{
neg.f16x2 r1313, r1312;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1315, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1316, {low, high};
}
{
neg.f16x2 r1317, r1316;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1319, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1320, {low, high};
}
{
add.f16x2 r1321, %69, %99;
}
{
add.f16x2 r1324, %59, r1321;
}
{
add.f16x2 r1327, %79, %89;
}
{
add.f16x2 r1330, r1324, r1327;
}
{
add.f16x2 r1333, %70, %100;
}
{
add.f16x2 r1336, %60, r1333;
}
{
add.f16x2 r1339, %80, %90;
}
{
add.f16x2 r1342, r1336, r1339;
}
{
add.f16x2 r1345, %69, %99;
}
{
mul.f16x2 r1348, r1345, r1311;
}
{
add.f16x2 r1351, %59, r1348;
}
{
add.f16x2 r1354, %79, %89;
}
{
mul.f16x2 r1357, r1354, r1315;
}
{
add.f16x2 r1360, r1351, r1357;
}
{
sub.f16x2 r1363, %70, %100;
}
{
mul.f16x2 r1366, r1363, r1313;
}
{
sub.f16x2 r1369, %80, %90;
}
{
mul.f16x2 r1372, r1369, r1317;
}
{
add.f16x2 r1375, r1366, r1372;
}
{
sub.f16x2 r1378, r1360, r1375;
}
{
add.f16x2 r1381, %69, %99;
}
{
mul.f16x2 r1384, r1381, r1311;
}
{
add.f16x2 r1387, %59, r1384;
}
{
add.f16x2 r1390, %79, %89;
}
{
mul.f16x2 r1393, r1390, r1315;
}
{
add.f16x2 r1396, r1387, r1393;
}
{
sub.f16x2 r1399, %70, %100;
}
{
mul.f16x2 r1402, r1399, r1313;
}
{
sub.f16x2 r1405, %80, %90;
}
{
mul.f16x2 r1408, r1405, r1317;
}
{
add.f16x2 r1411, r1402, r1408;
}
{
add.f16x2 r1414, r1396, r1411;
}
{
add.f16x2 r1417, %69, %99;
}
{
mul.f16x2 r1420, r1417, r1315;
}
{
add.f16x2 r1423, %59, r1420;
}
{
add.f16x2 r1426, %79, %89;
}
{
mul.f16x2 r1429, r1426, r1319;
}
{
add.f16x2 r1432, r1423, r1429;
}
{
sub.f16x2 r1435, %70, %100;
}
{
mul.f16x2 r1438, r1435, r1317;
}
{
sub.f16x2 r1441, %80, %90;
}
{
mul.f16x2 r1444, r1441, r1320;
}
{
add.f16x2 r1447, r1438, r1444;
}
{
sub.f16x2 r1450, r1432, r1447;
}
{
add.f16x2 r1453, %69, %99;
}
{
mul.f16x2 r1456, r1453, r1315;
}
{
add.f16x2 r1459, %59, r1456;
}
{
add.f16x2 r1462, %79, %89;
}
{
mul.f16x2 r1465, r1462, r1319;
}
{
add.f16x2 r1468, r1459, r1465;
}
{
sub.f16x2 r1471, %70, %100;
}
{
mul.f16x2 r1474, r1471, r1317;
}
{
sub.f16x2 r1477, %80, %90;
}
{
mul.f16x2 r1480, r1477, r1320;
}
{
add.f16x2 r1483, r1474, r1480;
}
{
add.f16x2 r1486, r1468, r1483;
}
{
add.f16x2 r1489, %70, %100;
}
{
mul.f16x2 r1492, r1489, r1311;
}
{
add.f16x2 r1495, %60, r1492;
}
{
add.f16x2 r1498, %80, %90;
}
{
mul.f16x2 r1501, r1498, r1315;
}
{
add.f16x2 r1504, r1495, r1501;
}
{
sub.f16x2 r1507, %69, %99;
}
{
mul.f16x2 r1510, r1507, r1313;
}
{
sub.f16x2 r1513, %79, %89;
}
{
mul.f16x2 r1516, r1513, r1317;
}
{
add.f16x2 r1519, r1510, r1516;
}
{
add.f16x2 r1522, r1504, r1519;
}
{
add.f16x2 r1525, %70, %100;
}
{
mul.f16x2 r1528, r1525, r1311;
}
{
add.f16x2 r1531, %60, r1528;
}
{
add.f16x2 r1534, %80, %90;
}
{
mul.f16x2 r1537, r1534, r1315;
}
{
add.f16x2 r1540, r1531, r1537;
}
{
sub.f16x2 r1543, %69, %99;
}
{
mul.f16x2 r1546, r1543, r1313;
}
{
sub.f16x2 r1549, %79, %89;
}
{
mul.f16x2 r1552, r1549, r1317;
}
{
add.f16x2 r1555, r1546, r1552;
}
{
sub.f16x2 r1558, r1540, r1555;
}
{
add.f16x2 r1561, %70, %100;
}
{
mul.f16x2 r1564, r1561, r1315;
}
{
add.f16x2 r1567, %60, r1564;
}
{
add.f16x2 r1570, %80, %90;
}
{
mul.f16x2 r1573, r1570, r1319;
}
{
add.f16x2 r1576, r1567, r1573;
}
{
sub.f16x2 r1579, %69, %99;
}
{
mul.f16x2 r1582, r1579, r1317;
}
{
sub.f16x2 r1585, %79, %89;
}
{
mul.f16x2 r1588, r1585, r1320;
}
{
add.f16x2 r1591, r1582, r1588;
}
{
add.f16x2 r1594, r1576, r1591;
}
{
add.f16x2 r1597, %70, %100;
}
{
mul.f16x2 r1600, r1597, r1315;
}
{
add.f16x2 r1603, %60, r1600;
}
{
add.f16x2 r1606, %80, %90;
}
{
mul.f16x2 r1609, r1606, r1319;
}
{
add.f16x2 r1612, r1603, r1609;
}
{
sub.f16x2 r1615, %69, %99;
}
{
mul.f16x2 r1618, r1615, r1317;
}
{
sub.f16x2 r1621, %79, %89;
}
{
mul.f16x2 r1624, r1621, r1320;
}
{
add.f16x2 r1627, r1618, r1624;
}
{
sub.f16x2 r1630, r1612, r1627;
}
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1633, {low, high};
}
mov.f32 f64, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1634, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1635, {low, high};
}
mov.f32 f68, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1636, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1637, {low, high};
}
mov.f32 f72, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1638, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1639, {low, high};
}
mov.f32 f76, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1640, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1643, {low, high};
}
mov.f32 f84, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1644, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1647, {low, high};
}
mov.f32 f92, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1648, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1649, {low, high};
}
mov.f32 f96, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1650, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1655, {low, high};
}
mov.f32 f108, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1656, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1663, {low, high};
}
mov.f32 f124, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1664, {low, high};
}
{
mul.f16x2 r1681, r412, r1633;
}
{
mul.f16x2 r1684, r556, r1634;
}
{
sub.f16x2 r1687, r1681, r1684;
}
{
mul.f16x2 r1690, r412, r1634;
}
{
fma.rn.f16x2 r1693, r556, r1633, r1690;
}
{
mul.f16x2 r1697, r734, r1635;
}
{
mul.f16x2 r1700, r878, r1636;
}
{
sub.f16x2 r1703, r1697, r1700;
}
{
mul.f16x2 r1706, r734, r1636;
}
{
fma.rn.f16x2 r1709, r878, r1635, r1706;
}
{
mul.f16x2 r1713, r1056, r1637;
}
{
mul.f16x2 r1716, r1200, r1638;
}
{
sub.f16x2 r1719, r1713, r1716;
}
{
mul.f16x2 r1722, r1056, r1638;
}
{
fma.rn.f16x2 r1725, r1200, r1637, r1722;
}
{
mul.f16x2 r1729, r1378, r1639;
}
{
mul.f16x2 r1732, r1522, r1640;
}
{
sub.f16x2 r1735, r1729, r1732;
}
{
mul.f16x2 r1738, r1378, r1640;
}
{
fma.rn.f16x2 r1741, r1522, r1639, r1738;
}
{
mul.f16x2 r1745, r484, r1635;
}
{
mul.f16x2 r1748, r628, r1636;
}
{
sub.f16x2 r1751, r1745, r1748;
}
{
mul.f16x2 r1754, r484, r1636;
}
{
fma.rn.f16x2 r1757, r628, r1635, r1754;
}
{
mul.f16x2 r1761, r806, r1639;
}
{
mul.f16x2 r1764, r950, r1640;
}
{
sub.f16x2 r1767, r1761, r1764;
}
{
mul.f16x2 r1770, r806, r1640;
}
{
fma.rn.f16x2 r1773, r950, r1639, r1770;
}
{
mul.f16x2 r1777, r1128, r1643;
}
{
mul.f16x2 r1780, r1272, r1644;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r1128, r1644;
}
{
fma.rn.f16x2 r1789, r1272, r1643, r1786;
}
{
mul.f16x2 r1793, r1450, r1647;
}
{
mul.f16x2 r1796, r1594, r1648;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r1450, r1648;
}
{
fma.rn.f16x2 r1805, r1594, r1647, r1802;
}
{
mul.f16x2 r1809, r520, r1637;
}
{
mul.f16x2 r1812, r664, r1638;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r520, r1638;
}
{
fma.rn.f16x2 r1821, r664, r1637, r1818;
}
{
mul.f16x2 r1825, r842, r1643;
}
{
mul.f16x2 r1828, r986, r1644;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r842, r1644;
}
{
fma.rn.f16x2 r1837, r986, r1643, r1834;
}
{
mul.f16x2 r1841, r1164, r1649;
}
{
mul.f16x2 r1844, r1308, r1650;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r1164, r1650;
}
{
fma.rn.f16x2 r1853, r1308, r1649, r1850;
}
{
mul.f16x2 r1857, r1486, r1655;
}
{
mul.f16x2 r1860, r1630, r1656;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r1486, r1656;
}
{
fma.rn.f16x2 r1869, r1630, r1655, r1866;
}
{
mul.f16x2 r1873, r448, r1639;
}
{
mul.f16x2 r1876, r592, r1640;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r448, r1640;
}
{
fma.rn.f16x2 r1885, r592, r1639, r1882;
}
{
mul.f16x2 r1889, r770, r1647;
}
{
mul.f16x2 r1892, r914, r1648;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r770, r1648;
}
{
fma.rn.f16x2 r1901, r914, r1647, r1898;
}
{
mul.f16x2 r1905, r1092, r1655;
}
{
mul.f16x2 r1908, r1236, r1656;
}
{
sub.f16x2 r1911, r1905, r1908;
}
{
mul.f16x2 r1914, r1092, r1656;
}
{
fma.rn.f16x2 r1917, r1236, r1655, r1914;
}
{
mul.f16x2 r1921, r1414, r1663;
}
{
mul.f16x2 r1924, r1558, r1664;
}
{
sub.f16x2 r1927, r1921, r1924;
}
{
mul.f16x2 r1930, r1414, r1664;
}
{
fma.rn.f16x2 r1933, r1558, r1663, r1930;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1937, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1938, {low, high};
}
{
neg.f16x2 r1939, r1938;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1942, {low, high};
}
{
neg.f16x2 r1943, r1942;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1946, {low, high};
}
{
add.f16x2 r1947, r364, r1330;
}
{
add.f16x2 r1950, r42, r1947;
}
{
add.f16x2 r1953, r686, r1008;
}
{
add.f16x2 r1956, r1950, r1953;
}
st.local.u32 [rd3], r1956;
{
add.f16x2 r1959, r376, r1342;
}
{
add.f16x2 r1962, r54, r1959;
}
{
add.f16x2 r1965, r698, r1020;
}
{
add.f16x2 r1968, r1962, r1965;
}
st.local.u32 [rd3+4], r1968;
{
add.f16x2 r1971, r364, r1330;
}
{
mul.f16x2 r1974, r1971, r1937;
}
{
add.f16x2 r1977, r42, r1974;
}
{
add.f16x2 r1980, r686, r1008;
}
{
mul.f16x2 r1983, r1980, r1941;
}
{
add.f16x2 r1986, r1977, r1983;
}
{
sub.f16x2 r1989, r376, r1342;
}
{
mul.f16x2 r1992, r1989, r1939;
}
{
sub.f16x2 r1995, r698, r1020;
}
{
mul.f16x2 r1998, r1995, r1943;
}
{
add.f16x2 r2001, r1992, r1998;
}
{
sub.f16x2 r2004, r1986, r2001;
}
st.local.u32 [rd3+40], r2004;
{
add.f16x2 r2007, r364, r1330;
}
{
mul.f16x2 r2010, r2007, r1937;
}
{
add.f16x2 r2013, r42, r2010;
}
{
add.f16x2 r2016, r686, r1008;
}
{
mul.f16x2 r2019, r2016, r1941;
}
{
add.f16x2 r2022, r2013, r2019;
}
{
sub.f16x2 r2025, r376, r1342;
}
{
mul.f16x2 r2028, r2025, r1939;
}
{
sub.f16x2 r2031, r698, r1020;
}
{
mul.f16x2 r2034, r2031, r1943;
}
{
add.f16x2 r2037, r2028, r2034;
}
{
add.f16x2 r2040, r2022, r2037;
}
st.local.u32 [rd3+160], r2040;
{
add.f16x2 r2043, r364, r1330;
}
{
mul.f16x2 r2046, r2043, r1941;
}
{
add.f16x2 r2049, r42, r2046;
}
{
add.f16x2 r2052, r686, r1008;
}
{
mul.f16x2 r2055, r2052, r1945;
}
{
add.f16x2 r2058, r2049, r2055;
}
{
sub.f16x2 r2061, r376, r1342;
}
{
mul.f16x2 r2064, r2061, r1943;
}
{
sub.f16x2 r2067, r698, r1020;
}
{
mul.f16x2 r2070, r2067, r1946;
}
{
add.f16x2 r2073, r2064, r2070;
}
{
sub.f16x2 r2076, r2058, r2073;
}
st.local.u32 [rd3+80], r2076;
{
add.f16x2 r2079, r364, r1330;
}
{
mul.f16x2 r2082, r2079, r1941;
}
{
add.f16x2 r2085, r42, r2082;
}
{
add.f16x2 r2088, r686, r1008;
}
{
mul.f16x2 r2091, r2088, r1945;
}
{
add.f16x2 r2094, r2085, r2091;
}
{
sub.f16x2 r2097, r376, r1342;
}
{
mul.f16x2 r2100, r2097, r1943;
}
{
sub.f16x2 r2103, r698, r1020;
}
{
mul.f16x2 r2106, r2103, r1946;
}
{
add.f16x2 r2109, r2100, r2106;
}
{
add.f16x2 r2112, r2094, r2109;
}
st.local.u32 [rd3+120], r2112;
{
add.f16x2 r2115, r376, r1342;
}
{
mul.f16x2 r2118, r2115, r1937;
}
{
add.f16x2 r2121, r54, r2118;
}
{
add.f16x2 r2124, r698, r1020;
}
{
mul.f16x2 r2127, r2124, r1941;
}
{
add.f16x2 r2130, r2121, r2127;
}
{
sub.f16x2 r2133, r364, r1330;
}
{
mul.f16x2 r2136, r2133, r1939;
}
{
sub.f16x2 r2139, r686, r1008;
}
{
mul.f16x2 r2142, r2139, r1943;
}
{
add.f16x2 r2145, r2136, r2142;
}
{
add.f16x2 r2148, r2130, r2145;
}
st.local.u32 [rd3+44], r2148;
{
add.f16x2 r2151, r376, r1342;
}
{
mul.f16x2 r2154, r2151, r1937;
}
{
add.f16x2 r2157, r54, r2154;
}
{
add.f16x2 r2160, r698, r1020;
}
{
mul.f16x2 r2163, r2160, r1941;
}
{
add.f16x2 r2166, r2157, r2163;
}
{
sub.f16x2 r2169, r364, r1330;
}
{
mul.f16x2 r2172, r2169, r1939;
}
{
sub.f16x2 r2175, r686, r1008;
}
{
mul.f16x2 r2178, r2175, r1943;
}
{
add.f16x2 r2181, r2172, r2178;
}
{
sub.f16x2 r2184, r2166, r2181;
}
st.local.u32 [rd3+164], r2184;
{
add.f16x2 r2187, r376, r1342;
}
{
mul.f16x2 r2190, r2187, r1941;
}
{
add.f16x2 r2193, r54, r2190;
}
{
add.f16x2 r2196, r698, r1020;
}
{
mul.f16x2 r2199, r2196, r1945;
}
{
add.f16x2 r2202, r2193, r2199;
}
{
sub.f16x2 r2205, r364, r1330;
}
{
mul.f16x2 r2208, r2205, r1943;
}
{
sub.f16x2 r2211, r686, r1008;
}
{
mul.f16x2 r2214, r2211, r1946;
}
{
add.f16x2 r2217, r2208, r2214;
}
{
add.f16x2 r2220, r2202, r2217;
}
st.local.u32 [rd3+84], r2220;
{
add.f16x2 r2223, r376, r1342;
}
{
mul.f16x2 r2226, r2223, r1941;
}
{
add.f16x2 r2229, r54, r2226;
}
{
add.f16x2 r2232, r698, r1020;
}
{
mul.f16x2 r2235, r2232, r1945;
}
{
add.f16x2 r2238, r2229, r2235;
}
{
sub.f16x2 r2241, r364, r1330;
}
{
mul.f16x2 r2244, r2241, r1943;
}
{
sub.f16x2 r2247, r686, r1008;
}
{
mul.f16x2 r2250, r2247, r1946;
}
{
add.f16x2 r2253, r2244, r2250;
}
{
sub.f16x2 r2256, r2238, r2253;
}
st.local.u32 [rd3+124], r2256;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2260, {low, high};
}
{
neg.f16x2 r2261, r2260;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2264, {low, high};
}
{
neg.f16x2 r2265, r2264;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2268, {low, high};
}
{
add.f16x2 r2269, r1687, r1735;
}
{
add.f16x2 r2272, r90, r2269;
}
{
add.f16x2 r2275, r1703, r1719;
}
{
add.f16x2 r11408, r2272, r2275;
}
st.local.u32 [rd3+8], r11408;
{
add.f16x2 r2281, r1693, r1741;
}
{
add.f16x2 r2284, r234, r2281;
}
{
add.f16x2 r2287, r1709, r1725;
}
{
add.f16x2 r2290, r2284, r2287;
}
st.local.u32 [rd3+12], r2290;
{
add.f16x2 r2293, r1687, r1735;
}
{
mul.f16x2 r2296, r2293, r2259;
}
{
add.f16x2 r2299, r90, r2296;
}
{
add.f16x2 r2302, r1703, r1719;
}
{
mul.f16x2 r2305, r2302, r2263;
}
{
add.f16x2 r2308, r2299, r2305;
}
{
sub.f16x2 r2311, r1693, r1741;
}
{
mul.f16x2 r2314, r2311, r2261;
}
{
sub.f16x2 r2317, r1709, r1725;
}
{
mul.f16x2 r2320, r2317, r2265;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
sub.f16x2 r2326, r2308, r2323;
}
st.local.u32 [rd3+48], r2326;
{
add.f16x2 r2329, r1687, r1735;
}
{
mul.f16x2 r2332, r2329, r2259;
}
{
add.f16x2 r2335, r90, r2332;
}
{
add.f16x2 r2338, r1703, r1719;
}
{
mul.f16x2 r2341, r2338, r2263;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
sub.f16x2 r2347, r1693, r1741;
}
{
mul.f16x2 r2350, r2347, r2261;
}
{
sub.f16x2 r2353, r1709, r1725;
}
{
mul.f16x2 r2356, r2353, r2265;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
add.f16x2 r2362, r2344, r2359;
}
st.local.u32 [rd3+168], r2362;
{
add.f16x2 r2365, r1687, r1735;
}
{
mul.f16x2 r2368, r2365, r2263;
}
{
add.f16x2 r2371, r90, r2368;
}
{
add.f16x2 r2374, r1703, r1719;
}
{
mul.f16x2 r2377, r2374, r2267;
}
{
add.f16x2 r2380, r2371, r2377;
}
{
sub.f16x2 r2383, r1693, r1741;
}
{
mul.f16x2 r2386, r2383, r2265;
}
{
sub.f16x2 r2389, r1709, r1725;
}
{
mul.f16x2 r2392, r2389, r2268;
}
{
add.f16x2 r2395, r2386, r2392;
}
{
sub.f16x2 r2398, r2380, r2395;
}
st.local.u32 [rd3+88], r2398;
{
add.f16x2 r2401, r1687, r1735;
}
{
mul.f16x2 r2404, r2401, r2263;
}
{
add.f16x2 r2407, r90, r2404;
}
{
add.f16x2 r2410, r1703, r1719;
}
{
mul.f16x2 r2413, r2410, r2267;
}
{
add.f16x2 r2416, r2407, r2413;
}
{
sub.f16x2 r2419, r1693, r1741;
}
{
mul.f16x2 r2422, r2419, r2265;
}
{
sub.f16x2 r2425, r1709, r1725;
}
{
mul.f16x2 r2428, r2425, r2268;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
add.f16x2 r2434, r2416, r2431;
}
st.local.u32 [rd3+128], r2434;
{
add.f16x2 r2437, r1693, r1741;
}
{
mul.f16x2 r2440, r2437, r2259;
}
{
add.f16x2 r2443, r234, r2440;
}
{
add.f16x2 r2446, r1709, r1725;
}
{
mul.f16x2 r2449, r2446, r2263;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
sub.f16x2 r2455, r1687, r1735;
}
{
mul.f16x2 r2458, r2455, r2261;
}
{
sub.f16x2 r2461, r1703, r1719;
}
{
mul.f16x2 r2464, r2461, r2265;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
add.f16x2 r2470, r2452, r2467;
}
st.local.u32 [rd3+52], r2470;
{
add.f16x2 r2473, r1693, r1741;
}
{
mul.f16x2 r2476, r2473, r2259;
}
{
add.f16x2 r2479, r234, r2476;
}
{
add.f16x2 r2482, r1709, r1725;
}
{
mul.f16x2 r2485, r2482, r2263;
}
{
add.f16x2 r2488, r2479, r2485;
}
{
sub.f16x2 r2491, r1687, r1735;
}
{
mul.f16x2 r2494, r2491, r2261;
}
{
sub.f16x2 r2497, r1703, r1719;
}
{
mul.f16x2 r2500, r2497, r2265;
}
{
add.f16x2 r2503, r2494, r2500;
}
{
sub.f16x2 r2506, r2488, r2503;
}
st.local.u32 [rd3+172], r2506;
{
add.f16x2 r2509, r1693, r1741;
}
{
mul.f16x2 r2512, r2509, r2263;
}
{
add.f16x2 r2515, r234, r2512;
}
{
add.f16x2 r2518, r1709, r1725;
}
{
mul.f16x2 r2521, r2518, r2267;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
sub.f16x2 r2527, r1687, r1735;
}
{
mul.f16x2 r2530, r2527, r2265;
}
{
sub.f16x2 r2533, r1703, r1719;
}
{
mul.f16x2 r2536, r2533, r2268;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
add.f16x2 r2542, r2524, r2539;
}
st.local.u32 [rd3+92], r2542;
{
add.f16x2 r2545, r1693, r1741;
}
{
mul.f16x2 r2548, r2545, r2263;
}
{
add.f16x2 r2551, r234, r2548;
}
{
add.f16x2 r2554, r1709, r1725;
}
{
mul.f16x2 r2557, r2554, r2267;
}
{
add.f16x2 r2560, r2551, r2557;
}
{
sub.f16x2 r2563, r1687, r1735;
}
{
mul.f16x2 r2566, r2563, r2265;
}
{
sub.f16x2 r2569, r1703, r1719;
}
{
mul.f16x2 r2572, r2569, r2268;
}
{
add.f16x2 r2575, r2566, r2572;
}
{
sub.f16x2 r2578, r2560, r2575;
}
st.local.u32 [rd3+132], r2578;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2582, {low, high};
}
{
neg.f16x2 r2583, r2582;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2585, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2586, {low, high};
}
{
neg.f16x2 r2587, r2586;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2590, {low, high};
}
{
add.f16x2 r2591, r1751, r1799;
}
{
add.f16x2 r2594, r162, r2591;
}
{
add.f16x2 r2597, r1767, r1783;
}
{
add.f16x2 r2600, r2594, r2597;
}
st.local.u32 [rd3+16], r2600;
{
add.f16x2 r2603, r1757, r1805;
}
{
add.f16x2 r2606, r306, r2603;
}
{
add.f16x2 r2609, r1773, r1789;
}
{
add.f16x2 r2612, r2606, r2609;
}
st.local.u32 [rd3+20], r2612;
{
add.f16x2 r2615, r1751, r1799;
}
{
mul.f16x2 r2618, r2615, r2581;
}
{
add.f16x2 r2621, r162, r2618;
}
{
add.f16x2 r2624, r1767, r1783;
}
{
mul.f16x2 r2627, r2624, r2585;
}
{
add.f16x2 r2630, r2621, r2627;
}
{
sub.f16x2 r2633, r1757, r1805;
}
{
mul.f16x2 r2636, r2633, r2583;
}
{
sub.f16x2 r2639, r1773, r1789;
}
{
mul.f16x2 r2642, r2639, r2587;
}
{
add.f16x2 r2645, r2636, r2642;
}
{
sub.f16x2 r2648, r2630, r2645;
}
st.local.u32 [rd3+56], r2648;
{
add.f16x2 r2651, r1751, r1799;
}
{
mul.f16x2 r2654, r2651, r2581;
}
{
add.f16x2 r2657, r162, r2654;
}
{
add.f16x2 r2660, r1767, r1783;
}
{
mul.f16x2 r2663, r2660, r2585;
}
{
add.f16x2 r2666, r2657, r2663;
}
{
sub.f16x2 r2669, r1757, r1805;
}
{
mul.f16x2 r2672, r2669, r2583;
}
{
sub.f16x2 r2675, r1773, r1789;
}
{
mul.f16x2 r2678, r2675, r2587;
}
{
add.f16x2 r2681, r2672, r2678;
}
{
add.f16x2 r2684, r2666, r2681;
}
st.local.u32 [rd3+176], r2684;
{
add.f16x2 r2687, r1751, r1799;
}
{
mul.f16x2 r2690, r2687, r2585;
}
{
add.f16x2 r2693, r162, r2690;
}
{
add.f16x2 r2696, r1767, r1783;
}
{
mul.f16x2 r2699, r2696, r2589;
}
{
add.f16x2 r2702, r2693, r2699;
}
{
sub.f16x2 r2705, r1757, r1805;
}
{
mul.f16x2 r2708, r2705, r2587;
}
{
sub.f16x2 r2711, r1773, r1789;
}
{
mul.f16x2 r2714, r2711, r2590;
}
{
add.f16x2 r2717, r2708, r2714;
}
{
sub.f16x2 r2720, r2702, r2717;
}
st.local.u32 [rd3+96], r2720;
{
add.f16x2 r2723, r1751, r1799;
}
{
mul.f16x2 r2726, r2723, r2585;
}
{
add.f16x2 r2729, r162, r2726;
}
{
add.f16x2 r2732, r1767, r1783;
}
{
mul.f16x2 r2735, r2732, r2589;
}
{
add.f16x2 r2738, r2729, r2735;
}
{
sub.f16x2 r2741, r1757, r1805;
}
{
mul.f16x2 r2744, r2741, r2587;
}
{
sub.f16x2 r2747, r1773, r1789;
}
{
mul.f16x2 r2750, r2747, r2590;
}
{
add.f16x2 r2753, r2744, r2750;
}
{
add.f16x2 r2756, r2738, r2753;
}
st.local.u32 [rd3+136], r2756;
{
add.f16x2 r2759, r1757, r1805;
}
{
mul.f16x2 r2762, r2759, r2581;
}
{
add.f16x2 r2765, r306, r2762;
}
{
add.f16x2 r2768, r1773, r1789;
}
{
mul.f16x2 r2771, r2768, r2585;
}
{
add.f16x2 r2774, r2765, r2771;
}
{
sub.f16x2 r2777, r1751, r1799;
}
{
mul.f16x2 r2780, r2777, r2583;
}
{
sub.f16x2 r2783, r1767, r1783;
}
{
mul.f16x2 r2786, r2783, r2587;
}
{
add.f16x2 r2789, r2780, r2786;
}
{
add.f16x2 r2792, r2774, r2789;
}
st.local.u32 [rd3+60], r2792;
{
add.f16x2 r2795, r1757, r1805;
}
{
mul.f16x2 r2798, r2795, r2581;
}
{
add.f16x2 r2801, r306, r2798;
}
{
add.f16x2 r2804, r1773, r1789;
}
{
mul.f16x2 r2807, r2804, r2585;
}
{
add.f16x2 r2810, r2801, r2807;
}
{
sub.f16x2 r2813, r1751, r1799;
}
{
mul.f16x2 r2816, r2813, r2583;
}
{
sub.f16x2 r2819, r1767, r1783;
}
{
mul.f16x2 r2822, r2819, r2587;
}
{
add.f16x2 r2825, r2816, r2822;
}
{
sub.f16x2 r2828, r2810, r2825;
}
st.local.u32 [rd3+180], r2828;
{
add.f16x2 r2831, r1757, r1805;
}
{
mul.f16x2 r2834, r2831, r2585;
}
{
add.f16x2 r2837, r306, r2834;
}
{
add.f16x2 r2840, r1773, r1789;
}
{
mul.f16x2 r2843, r2840, r2589;
}
{
add.f16x2 r2846, r2837, r2843;
}
{
sub.f16x2 r2849, r1751, r1799;
}
{
mul.f16x2 r2852, r2849, r2587;
}
{
sub.f16x2 r2855, r1767, r1783;
}
{
mul.f16x2 r2858, r2855, r2590;
}
{
add.f16x2 r2861, r2852, r2858;
}
{
add.f16x2 r2864, r2846, r2861;
}
st.local.u32 [rd3+100], r2864;
{
add.f16x2 r2867, r1757, r1805;
}
{
mul.f16x2 r2870, r2867, r2585;
}
{
add.f16x2 r2873, r306, r2870;
}
{
add.f16x2 r2876, r1773, r1789;
}
{
mul.f16x2 r2879, r2876, r2589;
}
{
add.f16x2 r2882, r2873, r2879;
}
{
sub.f16x2 r2885, r1751, r1799;
}
{
mul.f16x2 r2888, r2885, r2587;
}
{
sub.f16x2 r2891, r1767, r1783;
}
{
mul.f16x2 r2894, r2891, r2590;
}
{
add.f16x2 r2897, r2888, r2894;
}
{
sub.f16x2 r2900, r2882, r2897;
}
st.local.u32 [rd3+140], r2900;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2903, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2904, {low, high};
}
{
neg.f16x2 r2905, r2904;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2908, {low, high};
}
{
neg.f16x2 r2909, r2908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2911, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2912, {low, high};
}
{
add.f16x2 r2913, r1815, r1863;
}
{
add.f16x2 r2916, r198, r2913;
}
{
add.f16x2 r2919, r1831, r1847;
}
{
add.f16x2 r2922, r2916, r2919;
}
st.local.u32 [rd3+24], r2922;
{
add.f16x2 r2925, r1821, r1869;
}
{
add.f16x2 r2928, r342, r2925;
}
{
add.f16x2 r2931, r1837, r1853;
}
{
add.f16x2 r2934, r2928, r2931;
}
st.local.u32 [rd3+28], r2934;
{
add.f16x2 r2937, r1815, r1863;
}
{
mul.f16x2 r2940, r2937, r2903;
}
{
add.f16x2 r2943, r198, r2940;
}
{
add.f16x2 r2946, r1831, r1847;
}
{
mul.f16x2 r2949, r2946, r2907;
}
{
add.f16x2 r2952, r2943, r2949;
}
{
sub.f16x2 r2955, r1821, r1869;
}
{
mul.f16x2 r2958, r2955, r2905;
}
{
sub.f16x2 r2961, r1837, r1853;
}
{
mul.f16x2 r2964, r2961, r2909;
}
{
add.f16x2 r2967, r2958, r2964;
}
{
sub.f16x2 r2970, r2952, r2967;
}
st.local.u32 [rd3+64], r2970;
{
add.f16x2 r2973, r1815, r1863;
}
{
mul.f16x2 r2976, r2973, r2903;
}
{
add.f16x2 r2979, r198, r2976;
}
{
add.f16x2 r2982, r1831, r1847;
}
{
mul.f16x2 r2985, r2982, r2907;
}
{
add.f16x2 r2988, r2979, r2985;
}
{
sub.f16x2 r2991, r1821, r1869;
}
{
mul.f16x2 r2994, r2991, r2905;
}
{
sub.f16x2 r2997, r1837, r1853;
}
{
mul.f16x2 r3000, r2997, r2909;
}
{
add.f16x2 r3003, r2994, r3000;
}
{
add.f16x2 r3006, r2988, r3003;
}
st.local.u32 [rd3+184], r3006;
{
add.f16x2 r3009, r1815, r1863;
}
{
mul.f16x2 r3012, r3009, r2907;
}
{
add.f16x2 r3015, r198, r3012;
}
{
add.f16x2 r3018, r1831, r1847;
}
{
mul.f16x2 r3021, r3018, r2911;
}
{
add.f16x2 r3024, r3015, r3021;
}
{
sub.f16x2 r3027, r1821, r1869;
}
{
mul.f16x2 r3030, r3027, r2909;
}
{
sub.f16x2 r3033, r1837, r1853;
}
{
mul.f16x2 r3036, r3033, r2912;
}
{
add.f16x2 r3039, r3030, r3036;
}
{
sub.f16x2 r3042, r3024, r3039;
}
st.local.u32 [rd3+104], r3042;
{
add.f16x2 r3045, r1815, r1863;
}
{
mul.f16x2 r3048, r3045, r2907;
}
{
add.f16x2 r3051, r198, r3048;
}
{
add.f16x2 r3054, r1831, r1847;
}
{
mul.f16x2 r3057, r3054, r2911;
}
{
add.f16x2 r3060, r3051, r3057;
}
{
sub.f16x2 r3063, r1821, r1869;
}
{
mul.f16x2 r3066, r3063, r2909;
}
{
sub.f16x2 r3069, r1837, r1853;
}
{
mul.f16x2 r3072, r3069, r2912;
}
{
add.f16x2 r3075, r3066, r3072;
}
{
add.f16x2 r3078, r3060, r3075;
}
st.local.u32 [rd3+144], r3078;
{
add.f16x2 r3081, r1821, r1869;
}
{
mul.f16x2 r3084, r3081, r2903;
}
{
add.f16x2 r3087, r342, r3084;
}
{
add.f16x2 r3090, r1837, r1853;
}
{
mul.f16x2 r3093, r3090, r2907;
}
{
add.f16x2 r3096, r3087, r3093;
}
{
sub.f16x2 r3099, r1815, r1863;
}
{
mul.f16x2 r3102, r3099, r2905;
}
{
sub.f16x2 r3105, r1831, r1847;
}
{
mul.f16x2 r3108, r3105, r2909;
}
{
add.f16x2 r3111, r3102, r3108;
}
{
add.f16x2 r3114, r3096, r3111;
}
st.local.u32 [rd3+68], r3114;
{
add.f16x2 r3117, r1821, r1869;
}
{
mul.f16x2 r3120, r3117, r2903;
}
{
add.f16x2 r3123, r342, r3120;
}
{
add.f16x2 r3126, r1837, r1853;
}
{
mul.f16x2 r3129, r3126, r2907;
}
{
add.f16x2 r3132, r3123, r3129;
}
{
sub.f16x2 r3135, r1815, r1863;
}
{
mul.f16x2 r3138, r3135, r2905;
}
{
sub.f16x2 r3141, r1831, r1847;
}
{
mul.f16x2 r3144, r3141, r2909;
}
{
add.f16x2 r3147, r3138, r3144;
}
{
sub.f16x2 r3150, r3132, r3147;
}
st.local.u32 [rd3+188], r3150;
{
add.f16x2 r3153, r1821, r1869;
}
{
mul.f16x2 r3156, r3153, r2907;
}
{
add.f16x2 r3159, r342, r3156;
}
{
add.f16x2 r3162, r1837, r1853;
}
{
mul.f16x2 r3165, r3162, r2911;
}
{
add.f16x2 r3168, r3159, r3165;
}
{
sub.f16x2 r3171, r1815, r1863;
}
{
mul.f16x2 r3174, r3171, r2909;
}
{
sub.f16x2 r3177, r1831, r1847;
}
{
mul.f16x2 r3180, r3177, r2912;
}
{
add.f16x2 r3183, r3174, r3180;
}
{
add.f16x2 r3186, r3168, r3183;
}
st.local.u32 [rd3+108], r3186;
{
add.f16x2 r3189, r1821, r1869;
}
{
mul.f16x2 r3192, r3189, r2907;
}
{
add.f16x2 r3195, r342, r3192;
}
{
add.f16x2 r3198, r1837, r1853;
}
{
mul.f16x2 r3201, r3198, r2911;
}
{
add.f16x2 r3204, r3195, r3201;
}
{
sub.f16x2 r3207, r1815, r1863;
}
{
mul.f16x2 r3210, r3207, r2909;
}
{
sub.f16x2 r3213, r1831, r1847;
}
{
mul.f16x2 r3216, r3213, r2912;
}
{
add.f16x2 r3219, r3210, r3216;
}
{
sub.f16x2 r3222, r3204, r3219;
}
st.local.u32 [rd3+148], r3222;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3226, {low, high};
}
{
neg.f16x2 r3227, r3226;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3230, {low, high};
}
{
neg.f16x2 r3231, r3230;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3233, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3234, {low, high};
}
{
add.f16x2 r3235, r1879, r1927;
}
{
add.f16x2 r3238, r126, r3235;
}
{
add.f16x2 r3241, r1895, r1911;
}
{
add.f16x2 r3244, r3238, r3241;
}
st.local.u32 [rd3+32], r3244;
{
add.f16x2 r3247, r1885, r1933;
}
{
add.f16x2 r3250, r270, r3247;
}
{
add.f16x2 r3253, r1901, r1917;
}
{
add.f16x2 r3256, r3250, r3253;
}
st.local.u32 [rd3+36], r3256;
{
add.f16x2 r3259, r1879, r1927;
}
{
mul.f16x2 r3262, r3259, r3225;
}
{
add.f16x2 r3265, r126, r3262;
}
{
add.f16x2 r3268, r1895, r1911;
}
{
mul.f16x2 r3271, r3268, r3229;
}
{
add.f16x2 r3274, r3265, r3271;
}
{
sub.f16x2 r3277, r1885, r1933;
}
{
mul.f16x2 r3280, r3277, r3227;
}
{
sub.f16x2 r3283, r1901, r1917;
}
{
mul.f16x2 r3286, r3283, r3231;
}
{
add.f16x2 r3289, r3280, r3286;
}
{
sub.f16x2 r3292, r3274, r3289;
}
st.local.u32 [rd3+72], r3292;
{
add.f16x2 r3295, r1879, r1927;
}
{
mul.f16x2 r3298, r3295, r3225;
}
{
add.f16x2 r3301, r126, r3298;
}
{
add.f16x2 r3304, r1895, r1911;
}
{
mul.f16x2 r3307, r3304, r3229;
}
{
add.f16x2 r3310, r3301, r3307;
}
{
sub.f16x2 r3313, r1885, r1933;
}
{
mul.f16x2 r3316, r3313, r3227;
}
{
sub.f16x2 r3319, r1901, r1917;
}
{
mul.f16x2 r3322, r3319, r3231;
}
{
add.f16x2 r3325, r3316, r3322;
}
{
add.f16x2 r3328, r3310, r3325;
}
st.local.u32 [rd3+192], r3328;
{
add.f16x2 r3331, r1879, r1927;
}
{
mul.f16x2 r3334, r3331, r3229;
}
{
add.f16x2 r3337, r126, r3334;
}
{
add.f16x2 r3340, r1895, r1911;
}
{
mul.f16x2 r3343, r3340, r3233;
}
{
add.f16x2 r3346, r3337, r3343;
}
{
sub.f16x2 r3349, r1885, r1933;
}
{
mul.f16x2 r3352, r3349, r3231;
}
{
sub.f16x2 r3355, r1901, r1917;
}
{
mul.f16x2 r3358, r3355, r3234;
}
{
add.f16x2 r3361, r3352, r3358;
}
{
sub.f16x2 r3364, r3346, r3361;
}
st.local.u32 [rd3+112], r3364;
{
add.f16x2 r3367, r1879, r1927;
}
{
mul.f16x2 r3370, r3367, r3229;
}
{
add.f16x2 r3373, r126, r3370;
}
{
add.f16x2 r3376, r1895, r1911;
}
{
mul.f16x2 r3379, r3376, r3233;
}
{
add.f16x2 r3382, r3373, r3379;
}
{
sub.f16x2 r3385, r1885, r1933;
}
{
mul.f16x2 r3388, r3385, r3231;
}
{
sub.f16x2 r3391, r1901, r1917;
}
{
mul.f16x2 r3394, r3391, r3234;
}
{
add.f16x2 r3397, r3388, r3394;
}
{
add.f16x2 r3400, r3382, r3397;
}
st.local.u32 [rd3+152], r3400;
{
add.f16x2 r3403, r1885, r1933;
}
{
mul.f16x2 r3406, r3403, r3225;
}
{
add.f16x2 r3409, r270, r3406;
}
{
add.f16x2 r3412, r1901, r1917;
}
{
mul.f16x2 r3415, r3412, r3229;
}
{
add.f16x2 r3418, r3409, r3415;
}
{
sub.f16x2 r3421, r1879, r1927;
}
{
mul.f16x2 r3424, r3421, r3227;
}
{
sub.f16x2 r3427, r1895, r1911;
}
{
mul.f16x2 r3430, r3427, r3231;
}
{
add.f16x2 r3433, r3424, r3430;
}
{
add.f16x2 r3436, r3418, r3433;
}
st.local.u32 [rd3+76], r3436;
{
add.f16x2 r3439, r1885, r1933;
}
{
mul.f16x2 r3442, r3439, r3225;
}
{
add.f16x2 r3445, r270, r3442;
}
{
add.f16x2 r3448, r1901, r1917;
}
{
mul.f16x2 r3451, r3448, r3229;
}
{
add.f16x2 r3454, r3445, r3451;
}
{
sub.f16x2 r3457, r1879, r1927;
}
{
mul.f16x2 r3460, r3457, r3227;
}
{
sub.f16x2 r3463, r1895, r1911;
}
{
mul.f16x2 r3466, r3463, r3231;
}
{
add.f16x2 r3469, r3460, r3466;
}
{
sub.f16x2 r3472, r3454, r3469;
}
st.local.u32 [rd3+196], r3472;
{
add.f16x2 r3475, r1885, r1933;
}
{
mul.f16x2 r3478, r3475, r3229;
}
{
add.f16x2 r3481, r270, r3478;
}
{
add.f16x2 r3484, r1901, r1917;
}
{
mul.f16x2 r3487, r3484, r3233;
}
{
add.f16x2 r3490, r3481, r3487;
}
{
sub.f16x2 r3493, r1879, r1927;
}
{
mul.f16x2 r3496, r3493, r3231;
}
{
sub.f16x2 r3499, r1895, r1911;
}
{
mul.f16x2 r3502, r3499, r3234;
}
{
add.f16x2 r3505, r3496, r3502;
}
{
add.f16x2 r3508, r3490, r3505;
}
st.local.u32 [rd3+116], r3508;
{
add.f16x2 r3511, r1885, r1933;
}
{
mul.f16x2 r3514, r3511, r3229;
}
{
add.f16x2 r3517, r270, r3514;
}
{
add.f16x2 r3520, r1901, r1917;
}
{
mul.f16x2 r3523, r3520, r3233;
}
{
add.f16x2 r3526, r3517, r3523;
}
{
sub.f16x2 r3529, r1879, r1927;
}
{
mul.f16x2 r3532, r3529, r3231;
}
{
sub.f16x2 r3535, r1895, r1911;
}
{
mul.f16x2 r3538, r3535, r3234;
}
{
add.f16x2 r3541, r3532, r3538;
}
{
sub.f16x2 r3544, r3526, r3541;
}
st.local.u32 [rd3+156], r3544;
mov.u32 r3552, %tid.x;
mul.wide.u32 rd11, r3552, -776530087;
shr.u64 rd12, rd11, 41;
cvt.u32.u64 r4, rd12;
mul.lo.s32 r3553, r4, 625;
sub.s32 r3, r3552, r3553;
cvt.rn.f32.u32 f221, r3;
mul.f32 f222, f221, 0f39D2D427;
cos.approx.f32 f217, f222;
sin.approx.f32 f223, f222;
neg.f32 f218, f223;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r11410, {low, high};
}
mov.u32 r11409, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11410;
mov.b32 r3574, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11410;
mov.b32 r3576, {high, high};
}
bra.uni LBB0_1;
LBB0_2:
ld.local.u32 r11408, [rd5+60];
LBB0_1:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11410;
mov.b32 r3554, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11410;
mov.b32 r3556, {high, high};
}
mul.wide.u32 rd13, r11409, 8;
add.s64 rd14, rd3, rd13;
add.s64 rd5, rd14, 4;
ld.local.u32 r3559, [rd14+4];
{
mul.f16x2 r3558, r3559, r3556;
}
{
fma.rn.f16x2 r3561, r11408, r3554, r3558;
}
st.local.u32 [rd14], r3561;
{
mul.f16x2 r3565, r11408, r3556;
}
{
neg.f16x2 r3568, r3565;
}
{
fma.rn.f16x2 r3570, r3559, r3554, r3568;
}
st.local.u32 [rd14+4], r3570;
mov.f32 f238, 0fBF800000;
mov.f32 f239, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3578, {low, high};
}
{
mul.f16x2 r3579, r3576, r3578;
}
{
mul.f16x2 r3582, r11410, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11410;
mov.b32 r3585, {high, low};
}
{
fma.rn.f16x2 r3587, r3579, r3585, r3582;
}
ld.local.u32 r3603, [rd14+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3591, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3593, {high, high};
}
ld.local.u32 r3608, [rd14+12];
{
mul.f16x2 r3595, r3608, r3593;
}
{
fma.rn.f16x2 r3598, r3603, r3591, r3595;
}
st.local.u32 [rd14+8], r3598;
{
mul.f16x2 r3602, r3603, r3593;
}
{
neg.f16x2 r3605, r3602;
}
{
fma.rn.f16x2 r3607, r3608, r3591, r3605;
}
st.local.u32 [rd14+12], r3607;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3615, {low, high};
}
{
mul.f16x2 r3616, r3576, r3615;
}
{
mul.f16x2 r3619, r3587, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3622, {high, low};
}
{
fma.rn.f16x2 r3624, r3616, r3622, r3619;
}
ld.local.u32 r3640, [rd14+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3628, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3630, {high, high};
}
ld.local.u32 r3645, [rd14+20];
{
mul.f16x2 r3632, r3645, r3630;
}
{
fma.rn.f16x2 r3635, r3640, r3628, r3632;
}
st.local.u32 [rd14+16], r3635;
{
mul.f16x2 r3639, r3640, r3630;
}
{
neg.f16x2 r3642, r3639;
}
{
fma.rn.f16x2 r3644, r3645, r3628, r3642;
}
st.local.u32 [rd14+20], r3644;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3652, {low, high};
}
{
mul.f16x2 r3653, r3576, r3652;
}
{
mul.f16x2 r3656, r3624, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3659, {high, low};
}
{
fma.rn.f16x2 r3661, r3653, r3659, r3656;
}
ld.local.u32 r3677, [rd14+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3665, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3667, {high, high};
}
ld.local.u32 r3682, [rd14+28];
{
mul.f16x2 r3669, r3682, r3667;
}
{
fma.rn.f16x2 r3672, r3677, r3665, r3669;
}
st.local.u32 [rd14+24], r3672;
{
mul.f16x2 r3676, r3677, r3667;
}
{
neg.f16x2 r3679, r3676;
}
{
fma.rn.f16x2 r3681, r3682, r3665, r3679;
}
st.local.u32 [rd14+28], r3681;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3689, {low, high};
}
{
mul.f16x2 r3690, r3576, r3689;
}
{
mul.f16x2 r3693, r3661, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3696, {high, low};
}
{
fma.rn.f16x2 r3698, r3690, r3696, r3693;
}
ld.local.u32 r3710, [rd14+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3702, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3704, {high, high};
}
ld.local.u32 r3719, [rd14+36];
{
mul.f16x2 r3706, r3719, r3704;
}
{
fma.rn.f16x2 r3709, r3710, r3702, r3706;
}
st.local.u32 [rd14+32], r3709;
{
mul.f16x2 r3713, r3710, r3704;
}
{
neg.f16x2 r3716, r3713;
}
{
fma.rn.f16x2 r3718, r3719, r3702, r3716;
}
st.local.u32 [rd14+36], r3718;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3726, {low, high};
}
{
mul.f16x2 r3727, r3576, r3726;
}
{
mul.f16x2 r3730, r3698, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3733, {high, low};
}
{
fma.rn.f16x2 r3735, r3727, r3733, r3730;
}
ld.local.u32 r3747, [rd14+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3739, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3741, {high, high};
}
ld.local.u32 r3744, [rd14+44];
{
mul.f16x2 r3743, r3744, r3741;
}
{
fma.rn.f16x2 r3746, r3747, r3739, r3743;
}
st.local.u32 [rd14+40], r3746;
{
mul.f16x2 r3750, r3747, r3741;
}
{
neg.f16x2 r3753, r3750;
}
{
fma.rn.f16x2 r3755, r3744, r3739, r3753;
}
st.local.u32 [rd14+44], r3755;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3763, {low, high};
}
{
mul.f16x2 r3764, r3576, r3763;
}
{
mul.f16x2 r3767, r3735, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3770, {high, low};
}
{
fma.rn.f16x2 r3772, r3764, r3770, r3767;
}
ld.local.u32 r3784, [rd14+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3776, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3778, {high, high};
}
ld.local.u32 r3781, [rd14+52];
{
mul.f16x2 r3780, r3781, r3778;
}
{
fma.rn.f16x2 r3783, r3784, r3776, r3780;
}
st.local.u32 [rd14+48], r3783;
{
mul.f16x2 r3787, r3784, r3778;
}
{
neg.f16x2 r3790, r3787;
}
{
fma.rn.f16x2 r3792, r3781, r3776, r3790;
}
st.local.u32 [rd14+52], r3792;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3800, {low, high};
}
{
mul.f16x2 r3801, r3576, r3800;
}
{
mul.f16x2 r3804, r3772, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3807, {high, low};
}
{
fma.rn.f16x2 r3809, r3801, r3807, r3804;
}
ld.local.u32 r3821, [rd14+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3813, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3815, {high, high};
}
ld.local.u32 r3818, [rd14+60];
{
mul.f16x2 r3817, r3818, r3815;
}
{
fma.rn.f16x2 r3820, r3821, r3813, r3817;
}
st.local.u32 [rd14+56], r3820;
{
mul.f16x2 r3824, r3821, r3815;
}
{
neg.f16x2 r3827, r3824;
}
{
fma.rn.f16x2 r3829, r3818, r3813, r3827;
}
st.local.u32 [rd14+60], r3829;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3837, {low, high};
}
{
mul.f16x2 r3838, r3576, r3837;
}
{
mul.f16x2 r3841, r3809, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3844, {high, low};
}
{
fma.rn.f16x2 r11410, r3838, r3844, r3841;
}
add.s32 r11409, r11409, 8;
setp.eq.s32 p1, r11409, 25;
@p1 bra LBB0_3;
bra.uni LBB0_2;
LBB0_3:
shl.b32 r7378, r1, 3;
mov.u32 r7379, %50;
add.s32 r7380, r7379, r7378;
mad.lo.s32 r12, r4, 125000, r7380;
barrier.sync 0;
mad.lo.s32 r7381, r3, 200, r12;
ld.local.v2.u32 {r7382, r7383}, [rd3];
st.shared.v2.f32 [r7381], {r7382, r7383};
ld.local.v2.u32 {r7386, r7387}, [rd4+4];
st.shared.v2.f32 [r7381+8], {r7386, r7387};
ld.local.v2.u32 {r7390, r7391}, [rd4+12];
st.shared.v2.f32 [r7381+16], {r7390, r7391};
ld.local.v2.u32 {r7394, r7395}, [rd4+20];
st.shared.v2.f32 [r7381+24], {r7394, r7395};
ld.local.v2.u32 {r7398, r7399}, [rd4+28];
st.shared.v2.f32 [r7381+32], {r7398, r7399};
ld.local.v2.u32 {r7402, r7403}, [rd4+36];
st.shared.v2.f32 [r7381+40], {r7402, r7403};
ld.local.v2.u32 {r7406, r7407}, [rd4+44];
st.shared.v2.f32 [r7381+48], {r7406, r7407};
ld.local.v2.u32 {r7410, r7411}, [rd4+52];
st.shared.v2.f32 [r7381+56], {r7410, r7411};
ld.local.v2.u32 {r7414, r7415}, [rd4+60];
st.shared.v2.f32 [r7381+64], {r7414, r7415};
ld.local.v2.u32 {r7418, r7419}, [rd4+68];
st.shared.v2.f32 [r7381+72], {r7418, r7419};
ld.local.v2.u32 {r7422, r7423}, [rd4+76];
st.shared.v2.f32 [r7381+80], {r7422, r7423};
ld.local.v2.u32 {r7426, r7427}, [rd4+84];
st.shared.v2.f32 [r7381+88], {r7426, r7427};
ld.local.v2.u32 {r7430, r7431}, [rd4+92];
st.shared.v2.f32 [r7381+96], {r7430, r7431};
ld.local.v2.u32 {r7434, r7435}, [rd4+100];
st.shared.v2.f32 [r7381+104], {r7434, r7435};
ld.local.v2.u32 {r7438, r7439}, [rd4+108];
st.shared.v2.f32 [r7381+112], {r7438, r7439};
ld.local.v2.u32 {r7442, r7443}, [rd4+116];
st.shared.v2.f32 [r7381+120], {r7442, r7443};
ld.local.v2.u32 {r7446, r7447}, [rd4+124];
st.shared.v2.f32 [r7381+128], {r7446, r7447};
ld.local.v2.u32 {r7450, r7451}, [rd4+132];
st.shared.v2.f32 [r7381+136], {r7450, r7451};
ld.local.v2.u32 {r7454, r7455}, [rd4+140];
st.shared.v2.f32 [r7381+144], {r7454, r7455};
ld.local.v2.u32 {r7458, r7459}, [rd4+148];
st.shared.v2.f32 [r7381+152], {r7458, r7459};
ld.local.v2.u32 {r7462, r7463}, [rd4+156];
st.shared.v2.f32 [r7381+160], {r7462, r7463};
ld.local.v2.u32 {r7466, r7467}, [rd4+164];
st.shared.v2.f32 [r7381+168], {r7466, r7467};
ld.local.v2.u32 {r7470, r7471}, [rd4+172];
st.shared.v2.f32 [r7381+176], {r7470, r7471};
ld.local.v2.u32 {r7474, r7475}, [rd4+180];
st.shared.v2.f32 [r7381+184], {r7474, r7475};
ld.local.v2.u32 {r7478, r7479}, [rd4+188];
st.shared.v2.f32 [r7381+192], {r7478, r7479};
barrier.sync 0;
mad.lo.s32 r13, r3, -192, r7381;
ld.shared.u32 r3864, [r13];
ld.shared.u32 r3876, [r13+4];
ld.shared.u32 r4186, [r13+5000];
ld.shared.u32 r4198, [r13+5004];
ld.shared.u32 r4508, [r13+10000];
ld.shared.u32 r4520, [r13+10004];
ld.shared.u32 r4830, [r13+15000];
ld.shared.u32 r4842, [r13+15004];
ld.shared.u32 r5152, [r13+20000];
ld.shared.u32 r5164, [r13+20004];
ld.shared.u32 r3861, [r13+25000];
ld.shared.u32 r3873, [r13+25004];
ld.shared.u32 r4183, [r13+30000];
ld.shared.u32 r4195, [r13+30004];
ld.shared.u32 r4505, [r13+35000];
ld.shared.u32 r4517, [r13+35004];
ld.shared.u32 r4827, [r13+40000];
ld.shared.u32 r4839, [r13+40004];
ld.shared.u32 r5149, [r13+45000];
ld.shared.u32 r5161, [r13+45004];
ld.shared.u32 r3867, [r13+50000];
ld.shared.u32 r3879, [r13+50004];
ld.shared.u32 r4189, [r13+55000];
ld.shared.u32 r4201, [r13+55004];
ld.shared.u32 r4511, [r13+60000];
ld.shared.u32 r4523, [r13+60004];
ld.shared.u32 r4833, [r13+65000];
ld.shared.u32 r4845, [r13+65004];
ld.shared.u32 r5155, [r13+70000];
ld.shared.u32 r5167, [r13+70004];
ld.shared.u32 r3868, [r13+75000];
ld.shared.u32 r3880, [r13+75004];
ld.shared.u32 r4190, [r13+80000];
ld.shared.u32 r4202, [r13+80004];
ld.shared.u32 r4512, [r13+85000];
ld.shared.u32 r4524, [r13+85004];
ld.shared.u32 r4834, [r13+90000];
ld.shared.u32 r4846, [r13+90004];
ld.shared.u32 r5156, [r13+95000];
ld.shared.u32 r5168, [r13+95004];
ld.shared.u32 r3862, [r13+100000];
ld.shared.u32 r3874, [r13+100004];
ld.shared.u32 r4184, [r13+105000];
ld.shared.u32 r4196, [r13+105004];
ld.shared.u32 r4506, [r13+110000];
ld.shared.u32 r4518, [r13+110004];
ld.shared.u32 r4828, [r13+115000];
ld.shared.u32 r4840, [r13+115004];
ld.shared.u32 r5150, [r13+120000];
ld.shared.u32 r5162, [r13+120004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3850, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3851, {low, high};
}
{
neg.f16x2 r3852, r3851;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3854, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3855, {low, high};
}
{
neg.f16x2 r3856, r3855;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3859, {low, high};
}
{
add.f16x2 r3860, r3861, r3862;
}
{
add.f16x2 r3863, r3864, r3860;
}
{
add.f16x2 r3866, r3867, r3868;
}
{
add.f16x2 r3869, r3863, r3866;
}
{
add.f16x2 r3872, r3873, r3874;
}
{
add.f16x2 r3875, r3876, r3872;
}
{
add.f16x2 r3878, r3879, r3880;
}
{
add.f16x2 r3881, r3875, r3878;
}
{
add.f16x2 r3884, r3861, r3862;
}
{
mul.f16x2 r3887, r3884, r3850;
}
{
add.f16x2 r3890, r3864, r3887;
}
{
add.f16x2 r3893, r3867, r3868;
}
{
mul.f16x2 r3896, r3893, r3854;
}
{
add.f16x2 r3899, r3890, r3896;
}
{
sub.f16x2 r3902, r3873, r3874;
}
{
mul.f16x2 r3905, r3902, r3852;
}
{
sub.f16x2 r3908, r3879, r3880;
}
{
mul.f16x2 r3911, r3908, r3856;
}
{
add.f16x2 r3914, r3905, r3911;
}
{
sub.f16x2 r3917, r3899, r3914;
}
{
add.f16x2 r3920, r3861, r3862;
}
{
mul.f16x2 r3923, r3920, r3850;
}
{
add.f16x2 r3926, r3864, r3923;
}
{
add.f16x2 r3929, r3867, r3868;
}
{
mul.f16x2 r3932, r3929, r3854;
}
{
add.f16x2 r3935, r3926, r3932;
}
{
sub.f16x2 r3938, r3873, r3874;
}
{
mul.f16x2 r3941, r3938, r3852;
}
{
sub.f16x2 r3944, r3879, r3880;
}
{
mul.f16x2 r3947, r3944, r3856;
}
{
add.f16x2 r3950, r3941, r3947;
}
{
add.f16x2 r3953, r3935, r3950;
}
{
add.f16x2 r3956, r3861, r3862;
}
{
mul.f16x2 r3959, r3956, r3854;
}
{
add.f16x2 r3962, r3864, r3959;
}
{
add.f16x2 r3965, r3867, r3868;
}
{
mul.f16x2 r3968, r3965, r3858;
}
{
add.f16x2 r3971, r3962, r3968;
}
{
sub.f16x2 r3974, r3873, r3874;
}
{
mul.f16x2 r3977, r3974, r3856;
}
{
sub.f16x2 r3980, r3879, r3880;
}
{
mul.f16x2 r3983, r3980, r3859;
}
{
add.f16x2 r3986, r3977, r3983;
}
{
sub.f16x2 r3989, r3971, r3986;
}
{
add.f16x2 r3992, r3861, r3862;
}
{
mul.f16x2 r3995, r3992, r3854;
}
{
add.f16x2 r3998, r3864, r3995;
}
{
add.f16x2 r4001, r3867, r3868;
}
{
mul.f16x2 r4004, r4001, r3858;
}
{
add.f16x2 r4007, r3998, r4004;
}
{
sub.f16x2 r4010, r3873, r3874;
}
{
mul.f16x2 r4013, r4010, r3856;
}
{
sub.f16x2 r4016, r3879, r3880;
}
{
mul.f16x2 r4019, r4016, r3859;
}
{
add.f16x2 r4022, r4013, r4019;
}
{
add.f16x2 r4025, r4007, r4022;
}
{
add.f16x2 r4028, r3873, r3874;
}
{
mul.f16x2 r4031, r4028, r3850;
}
{
add.f16x2 r4034, r3876, r4031;
}
{
add.f16x2 r4037, r3879, r3880;
}
{
mul.f16x2 r4040, r4037, r3854;
}
{
add.f16x2 r4043, r4034, r4040;
}
{
sub.f16x2 r4046, r3861, r3862;
}
{
mul.f16x2 r4049, r4046, r3852;
}
{
sub.f16x2 r4052, r3867, r3868;
}
{
mul.f16x2 r4055, r4052, r3856;
}
{
add.f16x2 r4058, r4049, r4055;
}
{
add.f16x2 r4061, r4043, r4058;
}
{
add.f16x2 r4064, r3873, r3874;
}
{
mul.f16x2 r4067, r4064, r3850;
}
{
add.f16x2 r4070, r3876, r4067;
}
{
add.f16x2 r4073, r3879, r3880;
}
{
mul.f16x2 r4076, r4073, r3854;
}
{
add.f16x2 r4079, r4070, r4076;
}
{
sub.f16x2 r4082, r3861, r3862;
}
{
mul.f16x2 r4085, r4082, r3852;
}
{
sub.f16x2 r4088, r3867, r3868;
}
{
mul.f16x2 r4091, r4088, r3856;
}
{
add.f16x2 r4094, r4085, r4091;
}
{
sub.f16x2 r4097, r4079, r4094;
}
{
add.f16x2 r4100, r3873, r3874;
}
{
mul.f16x2 r4103, r4100, r3854;
}
{
add.f16x2 r4106, r3876, r4103;
}
{
add.f16x2 r4109, r3879, r3880;
}
{
mul.f16x2 r4112, r4109, r3858;
}
{
add.f16x2 r4115, r4106, r4112;
}
{
sub.f16x2 r4118, r3861, r3862;
}
{
mul.f16x2 r4121, r4118, r3856;
}
{
sub.f16x2 r4124, r3867, r3868;
}
{
mul.f16x2 r4127, r4124, r3859;
}
{
add.f16x2 r4130, r4121, r4127;
}
{
add.f16x2 r4133, r4115, r4130;
}
{
add.f16x2 r4136, r3873, r3874;
}
{
mul.f16x2 r4139, r4136, r3854;
}
{
add.f16x2 r4142, r3876, r4139;
}
{
add.f16x2 r4145, r3879, r3880;
}
{
mul.f16x2 r4148, r4145, r3858;
}
{
add.f16x2 r4151, r4142, r4148;
}
{
sub.f16x2 r4154, r3861, r3862;
}
{
mul.f16x2 r4157, r4154, r3856;
}
{
sub.f16x2 r4160, r3867, r3868;
}
{
mul.f16x2 r4163, r4160, r3859;
}
{
add.f16x2 r4166, r4157, r4163;
}
{
sub.f16x2 r4169, r4151, r4166;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4173, {low, high};
}
{
neg.f16x2 r4174, r4173;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4176, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4177, {low, high};
}
{
neg.f16x2 r4178, r4177;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4180, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4181, {low, high};
}
{
add.f16x2 r4182, r4183, r4184;
}
{
add.f16x2 r4185, r4186, r4182;
}
{
add.f16x2 r4188, r4189, r4190;
}
{
add.f16x2 r4191, r4185, r4188;
}
{
add.f16x2 r4194, r4195, r4196;
}
{
add.f16x2 r4197, r4198, r4194;
}
{
add.f16x2 r4200, r4201, r4202;
}
{
add.f16x2 r4203, r4197, r4200;
}
{
add.f16x2 r4206, r4183, r4184;
}
{
mul.f16x2 r4209, r4206, r4172;
}
{
add.f16x2 r4212, r4186, r4209;
}
{
add.f16x2 r4215, r4189, r4190;
}
{
mul.f16x2 r4218, r4215, r4176;
}
{
add.f16x2 r4221, r4212, r4218;
}
{
sub.f16x2 r4224, r4195, r4196;
}
{
mul.f16x2 r4227, r4224, r4174;
}
{
sub.f16x2 r4230, r4201, r4202;
}
{
mul.f16x2 r4233, r4230, r4178;
}
{
add.f16x2 r4236, r4227, r4233;
}
{
sub.f16x2 r4239, r4221, r4236;
}
{
add.f16x2 r4242, r4183, r4184;
}
{
mul.f16x2 r4245, r4242, r4172;
}
{
add.f16x2 r4248, r4186, r4245;
}
{
add.f16x2 r4251, r4189, r4190;
}
{
mul.f16x2 r4254, r4251, r4176;
}
{
add.f16x2 r4257, r4248, r4254;
}
{
sub.f16x2 r4260, r4195, r4196;
}
{
mul.f16x2 r4263, r4260, r4174;
}
{
sub.f16x2 r4266, r4201, r4202;
}
{
mul.f16x2 r4269, r4266, r4178;
}
{
add.f16x2 r4272, r4263, r4269;
}
{
add.f16x2 r4275, r4257, r4272;
}
{
add.f16x2 r4278, r4183, r4184;
}
{
mul.f16x2 r4281, r4278, r4176;
}
{
add.f16x2 r4284, r4186, r4281;
}
{
add.f16x2 r4287, r4189, r4190;
}
{
mul.f16x2 r4290, r4287, r4180;
}
{
add.f16x2 r4293, r4284, r4290;
}
{
sub.f16x2 r4296, r4195, r4196;
}
{
mul.f16x2 r4299, r4296, r4178;
}
{
sub.f16x2 r4302, r4201, r4202;
}
{
mul.f16x2 r4305, r4302, r4181;
}
{
add.f16x2 r4308, r4299, r4305;
}
{
sub.f16x2 r4311, r4293, r4308;
}
{
add.f16x2 r4314, r4183, r4184;
}
{
mul.f16x2 r4317, r4314, r4176;
}
{
add.f16x2 r4320, r4186, r4317;
}
{
add.f16x2 r4323, r4189, r4190;
}
{
mul.f16x2 r4326, r4323, r4180;
}
{
add.f16x2 r4329, r4320, r4326;
}
{
sub.f16x2 r4332, r4195, r4196;
}
{
mul.f16x2 r4335, r4332, r4178;
}
{
sub.f16x2 r4338, r4201, r4202;
}
{
mul.f16x2 r4341, r4338, r4181;
}
{
add.f16x2 r4344, r4335, r4341;
}
{
add.f16x2 r4347, r4329, r4344;
}
{
add.f16x2 r4350, r4195, r4196;
}
{
mul.f16x2 r4353, r4350, r4172;
}
{
add.f16x2 r4356, r4198, r4353;
}
{
add.f16x2 r4359, r4201, r4202;
}
{
mul.f16x2 r4362, r4359, r4176;
}
{
add.f16x2 r4365, r4356, r4362;
}
{
sub.f16x2 r4368, r4183, r4184;
}
{
mul.f16x2 r4371, r4368, r4174;
}
{
sub.f16x2 r4374, r4189, r4190;
}
{
mul.f16x2 r4377, r4374, r4178;
}
{
add.f16x2 r4380, r4371, r4377;
}
{
add.f16x2 r4383, r4365, r4380;
}
{
add.f16x2 r4386, r4195, r4196;
}
{
mul.f16x2 r4389, r4386, r4172;
}
{
add.f16x2 r4392, r4198, r4389;
}
{
add.f16x2 r4395, r4201, r4202;
}
{
mul.f16x2 r4398, r4395, r4176;
}
{
add.f16x2 r4401, r4392, r4398;
}
{
sub.f16x2 r4404, r4183, r4184;
}
{
mul.f16x2 r4407, r4404, r4174;
}
{
sub.f16x2 r4410, r4189, r4190;
}
{
mul.f16x2 r4413, r4410, r4178;
}
{
add.f16x2 r4416, r4407, r4413;
}
{
sub.f16x2 r4419, r4401, r4416;
}
{
add.f16x2 r4422, r4195, r4196;
}
{
mul.f16x2 r4425, r4422, r4176;
}
{
add.f16x2 r4428, r4198, r4425;
}
{
add.f16x2 r4431, r4201, r4202;
}
{
mul.f16x2 r4434, r4431, r4180;
}
{
add.f16x2 r4437, r4428, r4434;
}
{
sub.f16x2 r4440, r4183, r4184;
}
{
mul.f16x2 r4443, r4440, r4178;
}
{
sub.f16x2 r4446, r4189, r4190;
}
{
mul.f16x2 r4449, r4446, r4181;
}
{
add.f16x2 r4452, r4443, r4449;
}
{
add.f16x2 r4455, r4437, r4452;
}
{
add.f16x2 r4458, r4195, r4196;
}
{
mul.f16x2 r4461, r4458, r4176;
}
{
add.f16x2 r4464, r4198, r4461;
}
{
add.f16x2 r4467, r4201, r4202;
}
{
mul.f16x2 r4470, r4467, r4180;
}
{
add.f16x2 r4473, r4464, r4470;
}
{
sub.f16x2 r4476, r4183, r4184;
}
{
mul.f16x2 r4479, r4476, r4178;
}
{
sub.f16x2 r4482, r4189, r4190;
}
{
mul.f16x2 r4485, r4482, r4181;
}
{
add.f16x2 r4488, r4479, r4485;
}
{
sub.f16x2 r4491, r4473, r4488;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4494, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4495, {low, high};
}
{
neg.f16x2 r4496, r4495;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4499, {low, high};
}
{
neg.f16x2 r4500, r4499;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4502, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4503, {low, high};
}
{
add.f16x2 r4504, r4505, r4506;
}
{
add.f16x2 r4507, r4508, r4504;
}
{
add.f16x2 r4510, r4511, r4512;
}
{
add.f16x2 r4513, r4507, r4510;
}
{
add.f16x2 r4516, r4517, r4518;
}
{
add.f16x2 r4519, r4520, r4516;
}
{
add.f16x2 r4522, r4523, r4524;
}
{
add.f16x2 r4525, r4519, r4522;
}
{
add.f16x2 r4528, r4505, r4506;
}
{
mul.f16x2 r4531, r4528, r4494;
}
{
add.f16x2 r4534, r4508, r4531;
}
{
add.f16x2 r4537, r4511, r4512;
}
{
mul.f16x2 r4540, r4537, r4498;
}
{
add.f16x2 r4543, r4534, r4540;
}
{
sub.f16x2 r4546, r4517, r4518;
}
{
mul.f16x2 r4549, r4546, r4496;
}
{
sub.f16x2 r4552, r4523, r4524;
}
{
mul.f16x2 r4555, r4552, r4500;
}
{
add.f16x2 r4558, r4549, r4555;
}
{
sub.f16x2 r4561, r4543, r4558;
}
{
add.f16x2 r4564, r4505, r4506;
}
{
mul.f16x2 r4567, r4564, r4494;
}
{
add.f16x2 r4570, r4508, r4567;
}
{
add.f16x2 r4573, r4511, r4512;
}
{
mul.f16x2 r4576, r4573, r4498;
}
{
add.f16x2 r4579, r4570, r4576;
}
{
sub.f16x2 r4582, r4517, r4518;
}
{
mul.f16x2 r4585, r4582, r4496;
}
{
sub.f16x2 r4588, r4523, r4524;
}
{
mul.f16x2 r4591, r4588, r4500;
}
{
add.f16x2 r4594, r4585, r4591;
}
{
add.f16x2 r4597, r4579, r4594;
}
{
add.f16x2 r4600, r4505, r4506;
}
{
mul.f16x2 r4603, r4600, r4498;
}
{
add.f16x2 r4606, r4508, r4603;
}
{
add.f16x2 r4609, r4511, r4512;
}
{
mul.f16x2 r4612, r4609, r4502;
}
{
add.f16x2 r4615, r4606, r4612;
}
{
sub.f16x2 r4618, r4517, r4518;
}
{
mul.f16x2 r4621, r4618, r4500;
}
{
sub.f16x2 r4624, r4523, r4524;
}
{
mul.f16x2 r4627, r4624, r4503;
}
{
add.f16x2 r4630, r4621, r4627;
}
{
sub.f16x2 r4633, r4615, r4630;
}
{
add.f16x2 r4636, r4505, r4506;
}
{
mul.f16x2 r4639, r4636, r4498;
}
{
add.f16x2 r4642, r4508, r4639;
}
{
add.f16x2 r4645, r4511, r4512;
}
{
mul.f16x2 r4648, r4645, r4502;
}
{
add.f16x2 r4651, r4642, r4648;
}
{
sub.f16x2 r4654, r4517, r4518;
}
{
mul.f16x2 r4657, r4654, r4500;
}
{
sub.f16x2 r4660, r4523, r4524;
}
{
mul.f16x2 r4663, r4660, r4503;
}
{
add.f16x2 r4666, r4657, r4663;
}
{
add.f16x2 r4669, r4651, r4666;
}
{
add.f16x2 r4672, r4517, r4518;
}
{
mul.f16x2 r4675, r4672, r4494;
}
{
add.f16x2 r4678, r4520, r4675;
}
{
add.f16x2 r4681, r4523, r4524;
}
{
mul.f16x2 r4684, r4681, r4498;
}
{
add.f16x2 r4687, r4678, r4684;
}
{
sub.f16x2 r4690, r4505, r4506;
}
{
mul.f16x2 r4693, r4690, r4496;
}
{
sub.f16x2 r4696, r4511, r4512;
}
{
mul.f16x2 r4699, r4696, r4500;
}
{
add.f16x2 r4702, r4693, r4699;
}
{
add.f16x2 r4705, r4687, r4702;
}
{
add.f16x2 r4708, r4517, r4518;
}
{
mul.f16x2 r4711, r4708, r4494;
}
{
add.f16x2 r4714, r4520, r4711;
}
{
add.f16x2 r4717, r4523, r4524;
}
{
mul.f16x2 r4720, r4717, r4498;
}
{
add.f16x2 r4723, r4714, r4720;
}
{
sub.f16x2 r4726, r4505, r4506;
}
{
mul.f16x2 r4729, r4726, r4496;
}
{
sub.f16x2 r4732, r4511, r4512;
}
{
mul.f16x2 r4735, r4732, r4500;
}
{
add.f16x2 r4738, r4729, r4735;
}
{
sub.f16x2 r4741, r4723, r4738;
}
{
add.f16x2 r4744, r4517, r4518;
}
{
mul.f16x2 r4747, r4744, r4498;
}
{
add.f16x2 r4750, r4520, r4747;
}
{
add.f16x2 r4753, r4523, r4524;
}
{
mul.f16x2 r4756, r4753, r4502;
}
{
add.f16x2 r4759, r4750, r4756;
}
{
sub.f16x2 r4762, r4505, r4506;
}
{
mul.f16x2 r4765, r4762, r4500;
}
{
sub.f16x2 r4768, r4511, r4512;
}
{
mul.f16x2 r4771, r4768, r4503;
}
{
add.f16x2 r4774, r4765, r4771;
}
{
add.f16x2 r4777, r4759, r4774;
}
{
add.f16x2 r4780, r4517, r4518;
}
{
mul.f16x2 r4783, r4780, r4498;
}
{
add.f16x2 r4786, r4520, r4783;
}
{
add.f16x2 r4789, r4523, r4524;
}
{
mul.f16x2 r4792, r4789, r4502;
}
{
add.f16x2 r4795, r4786, r4792;
}
{
sub.f16x2 r4798, r4505, r4506;
}
{
mul.f16x2 r4801, r4798, r4500;
}
{
sub.f16x2 r4804, r4511, r4512;
}
{
mul.f16x2 r4807, r4804, r4503;
}
{
add.f16x2 r4810, r4801, r4807;
}
{
sub.f16x2 r4813, r4795, r4810;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4817, {low, high};
}
{
neg.f16x2 r4818, r4817;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4820, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4821, {low, high};
}
{
neg.f16x2 r4822, r4821;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4824, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4825, {low, high};
}
{
add.f16x2 r4826, r4827, r4828;
}
{
add.f16x2 r4829, r4830, r4826;
}
{
add.f16x2 r4832, r4833, r4834;
}
{
add.f16x2 r4835, r4829, r4832;
}
{
add.f16x2 r4838, r4839, r4840;
}
{
add.f16x2 r4841, r4842, r4838;
}
{
add.f16x2 r4844, r4845, r4846;
}
{
add.f16x2 r4847, r4841, r4844;
}
{
add.f16x2 r4850, r4827, r4828;
}
{
mul.f16x2 r4853, r4850, r4816;
}
{
add.f16x2 r4856, r4830, r4853;
}
{
add.f16x2 r4859, r4833, r4834;
}
{
mul.f16x2 r4862, r4859, r4820;
}
{
add.f16x2 r4865, r4856, r4862;
}
{
sub.f16x2 r4868, r4839, r4840;
}
{
mul.f16x2 r4871, r4868, r4818;
}
{
sub.f16x2 r4874, r4845, r4846;
}
{
mul.f16x2 r4877, r4874, r4822;
}
{
add.f16x2 r4880, r4871, r4877;
}
{
sub.f16x2 r4883, r4865, r4880;
}
{
add.f16x2 r4886, r4827, r4828;
}
{
mul.f16x2 r4889, r4886, r4816;
}
{
add.f16x2 r4892, r4830, r4889;
}
{
add.f16x2 r4895, r4833, r4834;
}
{
mul.f16x2 r4898, r4895, r4820;
}
{
add.f16x2 r4901, r4892, r4898;
}
{
sub.f16x2 r4904, r4839, r4840;
}
{
mul.f16x2 r4907, r4904, r4818;
}
{
sub.f16x2 r4910, r4845, r4846;
}
{
mul.f16x2 r4913, r4910, r4822;
}
{
add.f16x2 r4916, r4907, r4913;
}
{
add.f16x2 r4919, r4901, r4916;
}
{
add.f16x2 r4922, r4827, r4828;
}
{
mul.f16x2 r4925, r4922, r4820;
}
{
add.f16x2 r4928, r4830, r4925;
}
{
add.f16x2 r4931, r4833, r4834;
}
{
mul.f16x2 r4934, r4931, r4824;
}
{
add.f16x2 r4937, r4928, r4934;
}
{
sub.f16x2 r4940, r4839, r4840;
}
{
mul.f16x2 r4943, r4940, r4822;
}
{
sub.f16x2 r4946, r4845, r4846;
}
{
mul.f16x2 r4949, r4946, r4825;
}
{
add.f16x2 r4952, r4943, r4949;
}
{
sub.f16x2 r4955, r4937, r4952;
}
{
add.f16x2 r4958, r4827, r4828;
}
{
mul.f16x2 r4961, r4958, r4820;
}
{
add.f16x2 r4964, r4830, r4961;
}
{
add.f16x2 r4967, r4833, r4834;
}
{
mul.f16x2 r4970, r4967, r4824;
}
{
add.f16x2 r4973, r4964, r4970;
}
{
sub.f16x2 r4976, r4839, r4840;
}
{
mul.f16x2 r4979, r4976, r4822;
}
{
sub.f16x2 r4982, r4845, r4846;
}
{
mul.f16x2 r4985, r4982, r4825;
}
{
add.f16x2 r4988, r4979, r4985;
}
{
add.f16x2 r4991, r4973, r4988;
}
{
add.f16x2 r4994, r4839, r4840;
}
{
mul.f16x2 r4997, r4994, r4816;
}
{
add.f16x2 r5000, r4842, r4997;
}
{
add.f16x2 r5003, r4845, r4846;
}
{
mul.f16x2 r5006, r5003, r4820;
}
{
add.f16x2 r5009, r5000, r5006;
}
{
sub.f16x2 r5012, r4827, r4828;
}
{
mul.f16x2 r5015, r5012, r4818;
}
{
sub.f16x2 r5018, r4833, r4834;
}
{
mul.f16x2 r5021, r5018, r4822;
}
{
add.f16x2 r5024, r5015, r5021;
}
{
add.f16x2 r5027, r5009, r5024;
}
{
add.f16x2 r5030, r4839, r4840;
}
{
mul.f16x2 r5033, r5030, r4816;
}
{
add.f16x2 r5036, r4842, r5033;
}
{
add.f16x2 r5039, r4845, r4846;
}
{
mul.f16x2 r5042, r5039, r4820;
}
{
add.f16x2 r5045, r5036, r5042;
}
{
sub.f16x2 r5048, r4827, r4828;
}
{
mul.f16x2 r5051, r5048, r4818;
}
{
sub.f16x2 r5054, r4833, r4834;
}
{
mul.f16x2 r5057, r5054, r4822;
}
{
add.f16x2 r5060, r5051, r5057;
}
{
sub.f16x2 r5063, r5045, r5060;
}
{
add.f16x2 r5066, r4839, r4840;
}
{
mul.f16x2 r5069, r5066, r4820;
}
{
add.f16x2 r5072, r4842, r5069;
}
{
add.f16x2 r5075, r4845, r4846;
}
{
mul.f16x2 r5078, r5075, r4824;
}
{
add.f16x2 r5081, r5072, r5078;
}
{
sub.f16x2 r5084, r4827, r4828;
}
{
mul.f16x2 r5087, r5084, r4822;
}
{
sub.f16x2 r5090, r4833, r4834;
}
{
mul.f16x2 r5093, r5090, r4825;
}
{
add.f16x2 r5096, r5087, r5093;
}
{
add.f16x2 r5099, r5081, r5096;
}
{
add.f16x2 r5102, r4839, r4840;
}
{
mul.f16x2 r5105, r5102, r4820;
}
{
add.f16x2 r5108, r4842, r5105;
}
{
add.f16x2 r5111, r4845, r4846;
}
{
mul.f16x2 r5114, r5111, r4824;
}
{
add.f16x2 r5117, r5108, r5114;
}
{
sub.f16x2 r5120, r4827, r4828;
}
{
mul.f16x2 r5123, r5120, r4822;
}
{
sub.f16x2 r5126, r4833, r4834;
}
{
mul.f16x2 r5129, r5126, r4825;
}
{
add.f16x2 r5132, r5123, r5129;
}
{
sub.f16x2 r5135, r5117, r5132;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5138, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5139, {low, high};
}
{
neg.f16x2 r5140, r5139;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5142, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5143, {low, high};
}
{
neg.f16x2 r5144, r5143;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5146, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5147, {low, high};
}
{
add.f16x2 r5148, r5149, r5150;
}
{
add.f16x2 r5151, r5152, r5148;
}
{
add.f16x2 r5154, r5155, r5156;
}
{
add.f16x2 r5157, r5151, r5154;
}
{
add.f16x2 r5160, r5161, r5162;
}
{
add.f16x2 r5163, r5164, r5160;
}
{
add.f16x2 r5166, r5167, r5168;
}
{
add.f16x2 r5169, r5163, r5166;
}
{
add.f16x2 r5172, r5149, r5150;
}
{
mul.f16x2 r5175, r5172, r5138;
}
{
add.f16x2 r5178, r5152, r5175;
}
{
add.f16x2 r5181, r5155, r5156;
}
{
mul.f16x2 r5184, r5181, r5142;
}
{
add.f16x2 r5187, r5178, r5184;
}
{
sub.f16x2 r5190, r5161, r5162;
}
{
mul.f16x2 r5193, r5190, r5140;
}
{
sub.f16x2 r5196, r5167, r5168;
}
{
mul.f16x2 r5199, r5196, r5144;
}
{
add.f16x2 r5202, r5193, r5199;
}
{
sub.f16x2 r5205, r5187, r5202;
}
{
add.f16x2 r5208, r5149, r5150;
}
{
mul.f16x2 r5211, r5208, r5138;
}
{
add.f16x2 r5214, r5152, r5211;
}
{
add.f16x2 r5217, r5155, r5156;
}
{
mul.f16x2 r5220, r5217, r5142;
}
{
add.f16x2 r5223, r5214, r5220;
}
{
sub.f16x2 r5226, r5161, r5162;
}
{
mul.f16x2 r5229, r5226, r5140;
}
{
sub.f16x2 r5232, r5167, r5168;
}
{
mul.f16x2 r5235, r5232, r5144;
}
{
add.f16x2 r5238, r5229, r5235;
}
{
add.f16x2 r5241, r5223, r5238;
}
{
add.f16x2 r5244, r5149, r5150;
}
{
mul.f16x2 r5247, r5244, r5142;
}
{
add.f16x2 r5250, r5152, r5247;
}
{
add.f16x2 r5253, r5155, r5156;
}
{
mul.f16x2 r5256, r5253, r5146;
}
{
add.f16x2 r5259, r5250, r5256;
}
{
sub.f16x2 r5262, r5161, r5162;
}
{
mul.f16x2 r5265, r5262, r5144;
}
{
sub.f16x2 r5268, r5167, r5168;
}
{
mul.f16x2 r5271, r5268, r5147;
}
{
add.f16x2 r5274, r5265, r5271;
}
{
sub.f16x2 r5277, r5259, r5274;
}
{
add.f16x2 r5280, r5149, r5150;
}
{
mul.f16x2 r5283, r5280, r5142;
}
{
add.f16x2 r5286, r5152, r5283;
}
{
add.f16x2 r5289, r5155, r5156;
}
{
mul.f16x2 r5292, r5289, r5146;
}
{
add.f16x2 r5295, r5286, r5292;
}
{
sub.f16x2 r5298, r5161, r5162;
}
{
mul.f16x2 r5301, r5298, r5144;
}
{
sub.f16x2 r5304, r5167, r5168;
}
{
mul.f16x2 r5307, r5304, r5147;
}
{
add.f16x2 r5310, r5301, r5307;
}
{
add.f16x2 r5313, r5295, r5310;
}
{
add.f16x2 r5316, r5161, r5162;
}
{
mul.f16x2 r5319, r5316, r5138;
}
{
add.f16x2 r5322, r5164, r5319;
}
{
add.f16x2 r5325, r5167, r5168;
}
{
mul.f16x2 r5328, r5325, r5142;
}
{
add.f16x2 r5331, r5322, r5328;
}
{
sub.f16x2 r5334, r5149, r5150;
}
{
mul.f16x2 r5337, r5334, r5140;
}
{
sub.f16x2 r5340, r5155, r5156;
}
{
mul.f16x2 r5343, r5340, r5144;
}
{
add.f16x2 r5346, r5337, r5343;
}
{
add.f16x2 r5349, r5331, r5346;
}
{
add.f16x2 r5352, r5161, r5162;
}
{
mul.f16x2 r5355, r5352, r5138;
}
{
add.f16x2 r5358, r5164, r5355;
}
{
add.f16x2 r5361, r5167, r5168;
}
{
mul.f16x2 r5364, r5361, r5142;
}
{
add.f16x2 r5367, r5358, r5364;
}
{
sub.f16x2 r5370, r5149, r5150;
}
{
mul.f16x2 r5373, r5370, r5140;
}
{
sub.f16x2 r5376, r5155, r5156;
}
{
mul.f16x2 r5379, r5376, r5144;
}
{
add.f16x2 r5382, r5373, r5379;
}
{
sub.f16x2 r5385, r5367, r5382;
}
{
add.f16x2 r5388, r5161, r5162;
}
{
mul.f16x2 r5391, r5388, r5142;
}
{
add.f16x2 r5394, r5164, r5391;
}
{
add.f16x2 r5397, r5167, r5168;
}
{
mul.f16x2 r5400, r5397, r5146;
}
{
add.f16x2 r5403, r5394, r5400;
}
{
sub.f16x2 r5406, r5149, r5150;
}
{
mul.f16x2 r5409, r5406, r5144;
}
{
sub.f16x2 r5412, r5155, r5156;
}
{
mul.f16x2 r5415, r5412, r5147;
}
{
add.f16x2 r5418, r5409, r5415;
}
{
add.f16x2 r5421, r5403, r5418;
}
{
add.f16x2 r5424, r5161, r5162;
}
{
mul.f16x2 r5427, r5424, r5142;
}
{
add.f16x2 r5430, r5164, r5427;
}
{
add.f16x2 r5433, r5167, r5168;
}
{
mul.f16x2 r5436, r5433, r5146;
}
{
add.f16x2 r5439, r5430, r5436;
}
{
sub.f16x2 r5442, r5149, r5150;
}
{
mul.f16x2 r5445, r5442, r5144;
}
{
sub.f16x2 r5448, r5155, r5156;
}
{
mul.f16x2 r5451, r5448, r5147;
}
{
add.f16x2 r5454, r5445, r5451;
}
{
sub.f16x2 r5457, r5439, r5454;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r5460, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r5461, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r5462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r5463, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r5464, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r5465, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r5466, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r5467, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5470, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r5471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r5474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r5475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r5477, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r5483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r5491, {low, high};
}
{
mul.f16x2 r5508, r4239, r5460;
}
{
mul.f16x2 r5511, r4383, r5461;
}
{
sub.f16x2 r5514, r5508, r5511;
}
{
mul.f16x2 r5517, r4239, r5461;
}
{
fma.rn.f16x2 r5520, r4383, r5460, r5517;
}
{
mul.f16x2 r5524, r4561, r5462;
}
{
mul.f16x2 r5527, r4705, r5463;
}
{
sub.f16x2 r5530, r5524, r5527;
}
{
mul.f16x2 r5533, r4561, r5463;
}
{
fma.rn.f16x2 r5536, r4705, r5462, r5533;
}
{
mul.f16x2 r5540, r4883, r5464;
}
{
mul.f16x2 r5543, r5027, r5465;
}
{
sub.f16x2 r5546, r5540, r5543;
}
{
mul.f16x2 r5549, r4883, r5465;
}
{
fma.rn.f16x2 r5552, r5027, r5464, r5549;
}
{
mul.f16x2 r5556, r5205, r5466;
}
{
mul.f16x2 r5559, r5349, r5467;
}
{
sub.f16x2 r5562, r5556, r5559;
}
{
mul.f16x2 r5565, r5205, r5467;
}
{
fma.rn.f16x2 r5568, r5349, r5466, r5565;
}
{
mul.f16x2 r5572, r4311, r5462;
}
{
mul.f16x2 r5575, r4455, r5463;
}
{
sub.f16x2 r5578, r5572, r5575;
}
{
mul.f16x2 r5581, r4311, r5463;
}
{
fma.rn.f16x2 r5584, r4455, r5462, r5581;
}
{
mul.f16x2 r5588, r4633, r5466;
}
{
mul.f16x2 r5591, r4777, r5467;
}
{
sub.f16x2 r5594, r5588, r5591;
}
{
mul.f16x2 r5597, r4633, r5467;
}
{
fma.rn.f16x2 r5600, r4777, r5466, r5597;
}
{
mul.f16x2 r5604, r4955, r5470;
}
{
mul.f16x2 r5607, r5099, r5471;
}
{
sub.f16x2 r5610, r5604, r5607;
}
{
mul.f16x2 r5613, r4955, r5471;
}
{
fma.rn.f16x2 r5616, r5099, r5470, r5613;
}
{
mul.f16x2 r5620, r5277, r5474;
}
{
mul.f16x2 r5623, r5421, r5475;
}
{
sub.f16x2 r5626, r5620, r5623;
}
{
mul.f16x2 r5629, r5277, r5475;
}
{
fma.rn.f16x2 r5632, r5421, r5474, r5629;
}
{
mul.f16x2 r5636, r4347, r5464;
}
{
mul.f16x2 r5639, r4491, r5465;
}
{
sub.f16x2 r5642, r5636, r5639;
}
{
mul.f16x2 r5645, r4347, r5465;
}
{
fma.rn.f16x2 r5648, r4491, r5464, r5645;
}
{
mul.f16x2 r5652, r4669, r5470;
}
{
mul.f16x2 r5655, r4813, r5471;
}
{
sub.f16x2 r5658, r5652, r5655;
}
{
mul.f16x2 r5661, r4669, r5471;
}
{
fma.rn.f16x2 r5664, r4813, r5470, r5661;
}
{
mul.f16x2 r5668, r4991, r5476;
}
{
mul.f16x2 r5671, r5135, r5477;
}
{
sub.f16x2 r5674, r5668, r5671;
}
{
mul.f16x2 r5677, r4991, r5477;
}
{
fma.rn.f16x2 r5680, r5135, r5476, r5677;
}
{
mul.f16x2 r5684, r5313, r5482;
}
{
mul.f16x2 r5687, r5457, r5483;
}
{
sub.f16x2 r5690, r5684, r5687;
}
{
mul.f16x2 r5693, r5313, r5483;
}
{
fma.rn.f16x2 r5696, r5457, r5482, r5693;
}
{
mul.f16x2 r5700, r4275, r5466;
}
{
mul.f16x2 r5703, r4419, r5467;
}
{
sub.f16x2 r5706, r5700, r5703;
}
{
mul.f16x2 r5709, r4275, r5467;
}
{
fma.rn.f16x2 r5712, r4419, r5466, r5709;
}
{
mul.f16x2 r5716, r4597, r5474;
}
{
mul.f16x2 r5719, r4741, r5475;
}
{
sub.f16x2 r5722, r5716, r5719;
}
{
mul.f16x2 r5725, r4597, r5475;
}
{
fma.rn.f16x2 r5728, r4741, r5474, r5725;
}
{
mul.f16x2 r5732, r4919, r5482;
}
{
mul.f16x2 r5735, r5063, r5483;
}
{
sub.f16x2 r5738, r5732, r5735;
}
{
mul.f16x2 r5741, r4919, r5483;
}
{
fma.rn.f16x2 r5744, r5063, r5482, r5741;
}
{
mul.f16x2 r5748, r5241, r5490;
}
{
mul.f16x2 r5751, r5385, r5491;
}
{
sub.f16x2 r5754, r5748, r5751;
}
{
mul.f16x2 r5757, r5241, r5491;
}
{
fma.rn.f16x2 r5760, r5385, r5490, r5757;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5765, {low, high};
}
{
neg.f16x2 r5766, r5765;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5769, {low, high};
}
{
neg.f16x2 r5770, r5769;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5773, {low, high};
}
{
add.f16x2 r5774, r4191, r5157;
}
{
add.f16x2 r5777, r3869, r5774;
}
{
add.f16x2 r5780, r4513, r4835;
}
{
add.f16x2 r5783, r5777, r5780;
}
st.local.u32 [rd3], r5783;
{
add.f16x2 r5786, r4203, r5169;
}
{
add.f16x2 r5789, r3881, r5786;
}
{
add.f16x2 r5792, r4525, r4847;
}
{
add.f16x2 r5795, r5789, r5792;
}
st.local.u32 [rd4], r5795;
{
add.f16x2 r5798, r4191, r5157;
}
{
mul.f16x2 r5801, r5798, r5764;
}
{
add.f16x2 r5804, r3869, r5801;
}
{
add.f16x2 r5807, r4513, r4835;
}
{
mul.f16x2 r5810, r5807, r5768;
}
{
add.f16x2 r5813, r5804, r5810;
}
{
sub.f16x2 r5816, r4203, r5169;
}
{
mul.f16x2 r5819, r5816, r5766;
}
{
sub.f16x2 r5822, r4525, r4847;
}
{
mul.f16x2 r5825, r5822, r5770;
}
{
add.f16x2 r5828, r5819, r5825;
}
{
sub.f16x2 r5831, r5813, r5828;
}
st.local.u32 [rd4+36], r5831;
{
add.f16x2 r5834, r4191, r5157;
}
{
mul.f16x2 r5837, r5834, r5764;
}
{
add.f16x2 r5840, r3869, r5837;
}
{
add.f16x2 r5843, r4513, r4835;
}
{
mul.f16x2 r5846, r5843, r5768;
}
{
add.f16x2 r5849, r5840, r5846;
}
{
sub.f16x2 r5852, r4203, r5169;
}
{
mul.f16x2 r5855, r5852, r5766;
}
{
sub.f16x2 r5858, r4525, r4847;
}
{
mul.f16x2 r5861, r5858, r5770;
}
{
add.f16x2 r5864, r5855, r5861;
}
{
add.f16x2 r5867, r5849, r5864;
}
st.local.u32 [rd4+156], r5867;
{
add.f16x2 r5870, r4191, r5157;
}
{
mul.f16x2 r5873, r5870, r5768;
}
{
add.f16x2 r5876, r3869, r5873;
}
{
add.f16x2 r5879, r4513, r4835;
}
{
mul.f16x2 r5882, r5879, r5772;
}
{
add.f16x2 r5885, r5876, r5882;
}
{
sub.f16x2 r5888, r4203, r5169;
}
{
mul.f16x2 r5891, r5888, r5770;
}
{
sub.f16x2 r5894, r4525, r4847;
}
{
mul.f16x2 r5897, r5894, r5773;
}
{
add.f16x2 r5900, r5891, r5897;
}
{
sub.f16x2 r5903, r5885, r5900;
}
st.local.u32 [rd4+76], r5903;
{
add.f16x2 r5906, r4191, r5157;
}
{
mul.f16x2 r5909, r5906, r5768;
}
{
add.f16x2 r5912, r3869, r5909;
}
{
add.f16x2 r5915, r4513, r4835;
}
{
mul.f16x2 r5918, r5915, r5772;
}
{
add.f16x2 r5921, r5912, r5918;
}
{
sub.f16x2 r5924, r4203, r5169;
}
{
mul.f16x2 r5927, r5924, r5770;
}
{
sub.f16x2 r5930, r4525, r4847;
}
{
mul.f16x2 r5933, r5930, r5773;
}
{
add.f16x2 r5936, r5927, r5933;
}
{
add.f16x2 r5939, r5921, r5936;
}
st.local.u32 [rd4+116], r5939;
{
add.f16x2 r5942, r4203, r5169;
}
{
mul.f16x2 r5945, r5942, r5764;
}
{
add.f16x2 r5948, r3881, r5945;
}
{
add.f16x2 r5951, r4525, r4847;
}
{
mul.f16x2 r5954, r5951, r5768;
}
{
add.f16x2 r5957, r5948, r5954;
}
{
sub.f16x2 r5960, r4191, r5157;
}
{
mul.f16x2 r5963, r5960, r5766;
}
{
sub.f16x2 r5966, r4513, r4835;
}
{
mul.f16x2 r5969, r5966, r5770;
}
{
add.f16x2 r5972, r5963, r5969;
}
{
add.f16x2 r5975, r5957, r5972;
}
st.local.u32 [rd4+40], r5975;
{
add.f16x2 r5978, r4203, r5169;
}
{
mul.f16x2 r5981, r5978, r5764;
}
{
add.f16x2 r5984, r3881, r5981;
}
{
add.f16x2 r5987, r4525, r4847;
}
{
mul.f16x2 r5990, r5987, r5768;
}
{
add.f16x2 r5993, r5984, r5990;
}
{
sub.f16x2 r5996, r4191, r5157;
}
{
mul.f16x2 r5999, r5996, r5766;
}
{
sub.f16x2 r6002, r4513, r4835;
}
{
mul.f16x2 r6005, r6002, r5770;
}
{
add.f16x2 r6008, r5999, r6005;
}
{
sub.f16x2 r6011, r5993, r6008;
}
st.local.u32 [rd4+160], r6011;
{
add.f16x2 r6014, r4203, r5169;
}
{
mul.f16x2 r6017, r6014, r5768;
}
{
add.f16x2 r6020, r3881, r6017;
}
{
add.f16x2 r6023, r4525, r4847;
}
{
mul.f16x2 r6026, r6023, r5772;
}
{
add.f16x2 r6029, r6020, r6026;
}
{
sub.f16x2 r6032, r4191, r5157;
}
{
mul.f16x2 r6035, r6032, r5770;
}
{
sub.f16x2 r6038, r4513, r4835;
}
{
mul.f16x2 r6041, r6038, r5773;
}
{
add.f16x2 r6044, r6035, r6041;
}
{
add.f16x2 r6047, r6029, r6044;
}
st.local.u32 [rd4+80], r6047;
{
add.f16x2 r6050, r4203, r5169;
}
{
mul.f16x2 r6053, r6050, r5768;
}
{
add.f16x2 r6056, r3881, r6053;
}
{
add.f16x2 r6059, r4525, r4847;
}
{
mul.f16x2 r6062, r6059, r5772;
}
{
add.f16x2 r6065, r6056, r6062;
}
{
sub.f16x2 r6068, r4191, r5157;
}
{
mul.f16x2 r6071, r6068, r5770;
}
{
sub.f16x2 r6074, r4513, r4835;
}
{
mul.f16x2 r6077, r6074, r5773;
}
{
add.f16x2 r6080, r6071, r6077;
}
{
sub.f16x2 r6083, r6065, r6080;
}
st.local.u32 [rd4+120], r6083;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6086, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6087, {low, high};
}
{
neg.f16x2 r6088, r6087;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6090, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6091, {low, high};
}
{
neg.f16x2 r6092, r6091;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6095, {low, high};
}
{
add.f16x2 r6096, r5514, r5562;
}
{
add.f16x2 r6099, r3917, r6096;
}
{
add.f16x2 r6102, r5530, r5546;
}
{
add.f16x2 r11412, r6099, r6102;
}
st.local.u32 [rd4+4], r11412;
{
add.f16x2 r6108, r5520, r5568;
}
{
add.f16x2 r6111, r4061, r6108;
}
{
add.f16x2 r6114, r5536, r5552;
}
{
add.f16x2 r6117, r6111, r6114;
}
st.local.u32 [rd4+8], r6117;
{
add.f16x2 r6120, r5514, r5562;
}
{
mul.f16x2 r6123, r6120, r6086;
}
{
add.f16x2 r6126, r3917, r6123;
}
{
add.f16x2 r6129, r5530, r5546;
}
{
mul.f16x2 r6132, r6129, r6090;
}
{
add.f16x2 r6135, r6126, r6132;
}
{
sub.f16x2 r6138, r5520, r5568;
}
{
mul.f16x2 r6141, r6138, r6088;
}
{
sub.f16x2 r6144, r5536, r5552;
}
{
mul.f16x2 r6147, r6144, r6092;
}
{
add.f16x2 r6150, r6141, r6147;
}
{
sub.f16x2 r6153, r6135, r6150;
}
st.local.u32 [rd4+44], r6153;
{
add.f16x2 r6156, r5514, r5562;
}
{
mul.f16x2 r6159, r6156, r6086;
}
{
add.f16x2 r6162, r3917, r6159;
}
{
add.f16x2 r6165, r5530, r5546;
}
{
mul.f16x2 r6168, r6165, r6090;
}
{
add.f16x2 r6171, r6162, r6168;
}
{
sub.f16x2 r6174, r5520, r5568;
}
{
mul.f16x2 r6177, r6174, r6088;
}
{
sub.f16x2 r6180, r5536, r5552;
}
{
mul.f16x2 r6183, r6180, r6092;
}
{
add.f16x2 r6186, r6177, r6183;
}
{
add.f16x2 r6189, r6171, r6186;
}
st.local.u32 [rd4+164], r6189;
{
add.f16x2 r6192, r5514, r5562;
}
{
mul.f16x2 r6195, r6192, r6090;
}
{
add.f16x2 r6198, r3917, r6195;
}
{
add.f16x2 r6201, r5530, r5546;
}
{
mul.f16x2 r6204, r6201, r6094;
}
{
add.f16x2 r6207, r6198, r6204;
}
{
sub.f16x2 r6210, r5520, r5568;
}
{
mul.f16x2 r6213, r6210, r6092;
}
{
sub.f16x2 r6216, r5536, r5552;
}
{
mul.f16x2 r6219, r6216, r6095;
}
{
add.f16x2 r6222, r6213, r6219;
}
{
sub.f16x2 r6225, r6207, r6222;
}
st.local.u32 [rd4+84], r6225;
{
add.f16x2 r6228, r5514, r5562;
}
{
mul.f16x2 r6231, r6228, r6090;
}
{
add.f16x2 r6234, r3917, r6231;
}
{
add.f16x2 r6237, r5530, r5546;
}
{
mul.f16x2 r6240, r6237, r6094;
}
{
add.f16x2 r6243, r6234, r6240;
}
{
sub.f16x2 r6246, r5520, r5568;
}
{
mul.f16x2 r6249, r6246, r6092;
}
{
sub.f16x2 r6252, r5536, r5552;
}
{
mul.f16x2 r6255, r6252, r6095;
}
{
add.f16x2 r6258, r6249, r6255;
}
{
add.f16x2 r6261, r6243, r6258;
}
st.local.u32 [rd4+124], r6261;
{
add.f16x2 r6264, r5520, r5568;
}
{
mul.f16x2 r6267, r6264, r6086;
}
{
add.f16x2 r6270, r4061, r6267;
}
{
add.f16x2 r6273, r5536, r5552;
}
{
mul.f16x2 r6276, r6273, r6090;
}
{
add.f16x2 r6279, r6270, r6276;
}
{
sub.f16x2 r6282, r5514, r5562;
}
{
mul.f16x2 r6285, r6282, r6088;
}
{
sub.f16x2 r6288, r5530, r5546;
}
{
mul.f16x2 r6291, r6288, r6092;
}
{
add.f16x2 r6294, r6285, r6291;
}
{
add.f16x2 r6297, r6279, r6294;
}
st.local.u32 [rd4+48], r6297;
{
add.f16x2 r6300, r5520, r5568;
}
{
mul.f16x2 r6303, r6300, r6086;
}
{
add.f16x2 r6306, r4061, r6303;
}
{
add.f16x2 r6309, r5536, r5552;
}
{
mul.f16x2 r6312, r6309, r6090;
}
{
add.f16x2 r6315, r6306, r6312;
}
{
sub.f16x2 r6318, r5514, r5562;
}
{
mul.f16x2 r6321, r6318, r6088;
}
{
sub.f16x2 r6324, r5530, r5546;
}
{
mul.f16x2 r6327, r6324, r6092;
}
{
add.f16x2 r6330, r6321, r6327;
}
{
sub.f16x2 r6333, r6315, r6330;
}
st.local.u32 [rd4+168], r6333;
{
add.f16x2 r6336, r5520, r5568;
}
{
mul.f16x2 r6339, r6336, r6090;
}
{
add.f16x2 r6342, r4061, r6339;
}
{
add.f16x2 r6345, r5536, r5552;
}
{
mul.f16x2 r6348, r6345, r6094;
}
{
add.f16x2 r6351, r6342, r6348;
}
{
sub.f16x2 r6354, r5514, r5562;
}
{
mul.f16x2 r6357, r6354, r6092;
}
{
sub.f16x2 r6360, r5530, r5546;
}
{
mul.f16x2 r6363, r6360, r6095;
}
{
add.f16x2 r6366, r6357, r6363;
}
{
add.f16x2 r6369, r6351, r6366;
}
st.local.u32 [rd4+88], r6369;
{
add.f16x2 r6372, r5520, r5568;
}
{
mul.f16x2 r6375, r6372, r6090;
}
{
add.f16x2 r6378, r4061, r6375;
}
{
add.f16x2 r6381, r5536, r5552;
}
{
mul.f16x2 r6384, r6381, r6094;
}
{
add.f16x2 r6387, r6378, r6384;
}
{
sub.f16x2 r6390, r5514, r5562;
}
{
mul.f16x2 r6393, r6390, r6092;
}
{
sub.f16x2 r6396, r5530, r5546;
}
{
mul.f16x2 r6399, r6396, r6095;
}
{
add.f16x2 r6402, r6393, r6399;
}
{
sub.f16x2 r6405, r6387, r6402;
}
st.local.u32 [rd4+128], r6405;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6408, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6409, {low, high};
}
{
neg.f16x2 r6410, r6409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6412, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6413, {low, high};
}
{
neg.f16x2 r6414, r6413;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6417, {low, high};
}
{
add.f16x2 r6418, r5578, r5626;
}
{
add.f16x2 r6421, r3989, r6418;
}
{
add.f16x2 r6424, r5594, r5610;
}
{
add.f16x2 r6427, r6421, r6424;
}
st.local.u32 [rd4+12], r6427;
{
add.f16x2 r6430, r5584, r5632;
}
{
add.f16x2 r6433, r4133, r6430;
}
{
add.f16x2 r6436, r5600, r5616;
}
{
add.f16x2 r6439, r6433, r6436;
}
st.local.u32 [rd4+16], r6439;
{
add.f16x2 r6442, r5578, r5626;
}
{
mul.f16x2 r6445, r6442, r6408;
}
{
add.f16x2 r6448, r3989, r6445;
}
{
add.f16x2 r6451, r5594, r5610;
}
{
mul.f16x2 r6454, r6451, r6412;
}
{
add.f16x2 r6457, r6448, r6454;
}
{
sub.f16x2 r6460, r5584, r5632;
}
{
mul.f16x2 r6463, r6460, r6410;
}
{
sub.f16x2 r6466, r5600, r5616;
}
{
mul.f16x2 r6469, r6466, r6414;
}
{
add.f16x2 r6472, r6463, r6469;
}
{
sub.f16x2 r6475, r6457, r6472;
}
st.local.u32 [rd4+52], r6475;
{
add.f16x2 r6478, r5578, r5626;
}
{
mul.f16x2 r6481, r6478, r6408;
}
{
add.f16x2 r6484, r3989, r6481;
}
{
add.f16x2 r6487, r5594, r5610;
}
{
mul.f16x2 r6490, r6487, r6412;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
sub.f16x2 r6496, r5584, r5632;
}
{
mul.f16x2 r6499, r6496, r6410;
}
{
sub.f16x2 r6502, r5600, r5616;
}
{
mul.f16x2 r6505, r6502, r6414;
}
{
add.f16x2 r6508, r6499, r6505;
}
{
add.f16x2 r6511, r6493, r6508;
}
st.local.u32 [rd4+172], r6511;
{
add.f16x2 r6514, r5578, r5626;
}
{
mul.f16x2 r6517, r6514, r6412;
}
{
add.f16x2 r6520, r3989, r6517;
}
{
add.f16x2 r6523, r5594, r5610;
}
{
mul.f16x2 r6526, r6523, r6416;
}
{
add.f16x2 r6529, r6520, r6526;
}
{
sub.f16x2 r6532, r5584, r5632;
}
{
mul.f16x2 r6535, r6532, r6414;
}
{
sub.f16x2 r6538, r5600, r5616;
}
{
mul.f16x2 r6541, r6538, r6417;
}
{
add.f16x2 r6544, r6535, r6541;
}
{
sub.f16x2 r6547, r6529, r6544;
}
st.local.u32 [rd4+92], r6547;
{
add.f16x2 r6550, r5578, r5626;
}
{
mul.f16x2 r6553, r6550, r6412;
}
{
add.f16x2 r6556, r3989, r6553;
}
{
add.f16x2 r6559, r5594, r5610;
}
{
mul.f16x2 r6562, r6559, r6416;
}
{
add.f16x2 r6565, r6556, r6562;
}
{
sub.f16x2 r6568, r5584, r5632;
}
{
mul.f16x2 r6571, r6568, r6414;
}
{
sub.f16x2 r6574, r5600, r5616;
}
{
mul.f16x2 r6577, r6574, r6417;
}
{
add.f16x2 r6580, r6571, r6577;
}
{
add.f16x2 r6583, r6565, r6580;
}
st.local.u32 [rd4+132], r6583;
{
add.f16x2 r6586, r5584, r5632;
}
{
mul.f16x2 r6589, r6586, r6408;
}
{
add.f16x2 r6592, r4133, r6589;
}
{
add.f16x2 r6595, r5600, r5616;
}
{
mul.f16x2 r6598, r6595, r6412;
}
{
add.f16x2 r6601, r6592, r6598;
}
{
sub.f16x2 r6604, r5578, r5626;
}
{
mul.f16x2 r6607, r6604, r6410;
}
{
sub.f16x2 r6610, r5594, r5610;
}
{
mul.f16x2 r6613, r6610, r6414;
}
{
add.f16x2 r6616, r6607, r6613;
}
{
add.f16x2 r6619, r6601, r6616;
}
st.local.u32 [rd4+56], r6619;
{
add.f16x2 r6622, r5584, r5632;
}
{
mul.f16x2 r6625, r6622, r6408;
}
{
add.f16x2 r6628, r4133, r6625;
}
{
add.f16x2 r6631, r5600, r5616;
}
{
mul.f16x2 r6634, r6631, r6412;
}
{
add.f16x2 r6637, r6628, r6634;
}
{
sub.f16x2 r6640, r5578, r5626;
}
{
mul.f16x2 r6643, r6640, r6410;
}
{
sub.f16x2 r6646, r5594, r5610;
}
{
mul.f16x2 r6649, r6646, r6414;
}
{
add.f16x2 r6652, r6643, r6649;
}
{
sub.f16x2 r6655, r6637, r6652;
}
st.local.u32 [rd4+176], r6655;
{
add.f16x2 r6658, r5584, r5632;
}
{
mul.f16x2 r6661, r6658, r6412;
}
{
add.f16x2 r6664, r4133, r6661;
}
{
add.f16x2 r6667, r5600, r5616;
}
{
mul.f16x2 r6670, r6667, r6416;
}
{
add.f16x2 r6673, r6664, r6670;
}
{
sub.f16x2 r6676, r5578, r5626;
}
{
mul.f16x2 r6679, r6676, r6414;
}
{
sub.f16x2 r6682, r5594, r5610;
}
{
mul.f16x2 r6685, r6682, r6417;
}
{
add.f16x2 r6688, r6679, r6685;
}
{
add.f16x2 r6691, r6673, r6688;
}
st.local.u32 [rd4+96], r6691;
{
add.f16x2 r6694, r5584, r5632;
}
{
mul.f16x2 r6697, r6694, r6412;
}
{
add.f16x2 r6700, r4133, r6697;
}
{
add.f16x2 r6703, r5600, r5616;
}
{
mul.f16x2 r6706, r6703, r6416;
}
{
add.f16x2 r6709, r6700, r6706;
}
{
sub.f16x2 r6712, r5578, r5626;
}
{
mul.f16x2 r6715, r6712, r6414;
}
{
sub.f16x2 r6718, r5594, r5610;
}
{
mul.f16x2 r6721, r6718, r6417;
}
{
add.f16x2 r6724, r6715, r6721;
}
{
sub.f16x2 r6727, r6709, r6724;
}
st.local.u32 [rd4+136], r6727;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6730, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6731, {low, high};
}
{
neg.f16x2 r6732, r6731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6735, {low, high};
}
{
neg.f16x2 r6736, r6735;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6739, {low, high};
}
{
add.f16x2 r6740, r5642, r5690;
}
{
add.f16x2 r6743, r4025, r6740;
}
{
add.f16x2 r6746, r5658, r5674;
}
{
add.f16x2 r6749, r6743, r6746;
}
st.local.u32 [rd4+20], r6749;
{
add.f16x2 r6752, r5648, r5696;
}
{
add.f16x2 r6755, r4169, r6752;
}
{
add.f16x2 r6758, r5664, r5680;
}
{
add.f16x2 r6761, r6755, r6758;
}
st.local.u32 [rd4+24], r6761;
{
add.f16x2 r6764, r5642, r5690;
}
{
mul.f16x2 r6767, r6764, r6730;
}
{
add.f16x2 r6770, r4025, r6767;
}
{
add.f16x2 r6773, r5658, r5674;
}
{
mul.f16x2 r6776, r6773, r6734;
}
{
add.f16x2 r6779, r6770, r6776;
}
{
sub.f16x2 r6782, r5648, r5696;
}
{
mul.f16x2 r6785, r6782, r6732;
}
{
sub.f16x2 r6788, r5664, r5680;
}
{
mul.f16x2 r6791, r6788, r6736;
}
{
add.f16x2 r6794, r6785, r6791;
}
{
sub.f16x2 r6797, r6779, r6794;
}
st.local.u32 [rd4+60], r6797;
{
add.f16x2 r6800, r5642, r5690;
}
{
mul.f16x2 r6803, r6800, r6730;
}
{
add.f16x2 r6806, r4025, r6803;
}
{
add.f16x2 r6809, r5658, r5674;
}
{
mul.f16x2 r6812, r6809, r6734;
}
{
add.f16x2 r6815, r6806, r6812;
}
{
sub.f16x2 r6818, r5648, r5696;
}
{
mul.f16x2 r6821, r6818, r6732;
}
{
sub.f16x2 r6824, r5664, r5680;
}
{
mul.f16x2 r6827, r6824, r6736;
}
{
add.f16x2 r6830, r6821, r6827;
}
{
add.f16x2 r6833, r6815, r6830;
}
st.local.u32 [rd4+180], r6833;
{
add.f16x2 r6836, r5642, r5690;
}
{
mul.f16x2 r6839, r6836, r6734;
}
{
add.f16x2 r6842, r4025, r6839;
}
{
add.f16x2 r6845, r5658, r5674;
}
{
mul.f16x2 r6848, r6845, r6738;
}
{
add.f16x2 r6851, r6842, r6848;
}
{
sub.f16x2 r6854, r5648, r5696;
}
{
mul.f16x2 r6857, r6854, r6736;
}
{
sub.f16x2 r6860, r5664, r5680;
}
{
mul.f16x2 r6863, r6860, r6739;
}
{
add.f16x2 r6866, r6857, r6863;
}
{
sub.f16x2 r6869, r6851, r6866;
}
st.local.u32 [rd4+100], r6869;
{
add.f16x2 r6872, r5642, r5690;
}
{
mul.f16x2 r6875, r6872, r6734;
}
{
add.f16x2 r6878, r4025, r6875;
}
{
add.f16x2 r6881, r5658, r5674;
}
{
mul.f16x2 r6884, r6881, r6738;
}
{
add.f16x2 r6887, r6878, r6884;
}
{
sub.f16x2 r6890, r5648, r5696;
}
{
mul.f16x2 r6893, r6890, r6736;
}
{
sub.f16x2 r6896, r5664, r5680;
}
{
mul.f16x2 r6899, r6896, r6739;
}
{
add.f16x2 r6902, r6893, r6899;
}
{
add.f16x2 r6905, r6887, r6902;
}
st.local.u32 [rd4+140], r6905;
{
add.f16x2 r6908, r5648, r5696;
}
{
mul.f16x2 r6911, r6908, r6730;
}
{
add.f16x2 r6914, r4169, r6911;
}
{
add.f16x2 r6917, r5664, r5680;
}
{
mul.f16x2 r6920, r6917, r6734;
}
{
add.f16x2 r6923, r6914, r6920;
}
{
sub.f16x2 r6926, r5642, r5690;
}
{
mul.f16x2 r6929, r6926, r6732;
}
{
sub.f16x2 r6932, r5658, r5674;
}
{
mul.f16x2 r6935, r6932, r6736;
}
{
add.f16x2 r6938, r6929, r6935;
}
{
add.f16x2 r6941, r6923, r6938;
}
st.local.u32 [rd4+64], r6941;
{
add.f16x2 r6944, r5648, r5696;
}
{
mul.f16x2 r6947, r6944, r6730;
}
{
add.f16x2 r6950, r4169, r6947;
}
{
add.f16x2 r6953, r5664, r5680;
}
{
mul.f16x2 r6956, r6953, r6734;
}
{
add.f16x2 r6959, r6950, r6956;
}
{
sub.f16x2 r6962, r5642, r5690;
}
{
mul.f16x2 r6965, r6962, r6732;
}
{
sub.f16x2 r6968, r5658, r5674;
}
{
mul.f16x2 r6971, r6968, r6736;
}
{
add.f16x2 r6974, r6965, r6971;
}
{
sub.f16x2 r6977, r6959, r6974;
}
st.local.u32 [rd4+184], r6977;
{
add.f16x2 r6980, r5648, r5696;
}
{
mul.f16x2 r6983, r6980, r6734;
}
{
add.f16x2 r6986, r4169, r6983;
}
{
add.f16x2 r6989, r5664, r5680;
}
{
mul.f16x2 r6992, r6989, r6738;
}
{
add.f16x2 r6995, r6986, r6992;
}
{
sub.f16x2 r6998, r5642, r5690;
}
{
mul.f16x2 r7001, r6998, r6736;
}
{
sub.f16x2 r7004, r5658, r5674;
}
{
mul.f16x2 r7007, r7004, r6739;
}
{
add.f16x2 r7010, r7001, r7007;
}
{
add.f16x2 r7013, r6995, r7010;
}
st.local.u32 [rd4+104], r7013;
{
add.f16x2 r7016, r5648, r5696;
}
{
mul.f16x2 r7019, r7016, r6734;
}
{
add.f16x2 r7022, r4169, r7019;
}
{
add.f16x2 r7025, r5664, r5680;
}
{
mul.f16x2 r7028, r7025, r6738;
}
{
add.f16x2 r7031, r7022, r7028;
}
{
sub.f16x2 r7034, r5642, r5690;
}
{
mul.f16x2 r7037, r7034, r6736;
}
{
sub.f16x2 r7040, r5658, r5674;
}
{
mul.f16x2 r7043, r7040, r6739;
}
{
add.f16x2 r7046, r7037, r7043;
}
{
sub.f16x2 r7049, r7031, r7046;
}
st.local.u32 [rd4+144], r7049;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7052, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7053, {low, high};
}
{
neg.f16x2 r7054, r7053;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7057, {low, high};
}
{
neg.f16x2 r7058, r7057;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7061, {low, high};
}
{
add.f16x2 r7062, r5706, r5754;
}
{
add.f16x2 r7065, r3953, r7062;
}
{
add.f16x2 r7068, r5722, r5738;
}
{
add.f16x2 r7071, r7065, r7068;
}
st.local.u32 [rd4+28], r7071;
{
add.f16x2 r7074, r5712, r5760;
}
{
add.f16x2 r7077, r4097, r7074;
}
{
add.f16x2 r7080, r5728, r5744;
}
{
add.f16x2 r7083, r7077, r7080;
}
st.local.u32 [rd4+32], r7083;
{
add.f16x2 r7086, r5706, r5754;
}
{
mul.f16x2 r7089, r7086, r7052;
}
{
add.f16x2 r7092, r3953, r7089;
}
{
add.f16x2 r7095, r5722, r5738;
}
{
mul.f16x2 r7098, r7095, r7056;
}
{
add.f16x2 r7101, r7092, r7098;
}
{
sub.f16x2 r7104, r5712, r5760;
}
{
mul.f16x2 r7107, r7104, r7054;
}
{
sub.f16x2 r7110, r5728, r5744;
}
{
mul.f16x2 r7113, r7110, r7058;
}
{
add.f16x2 r7116, r7107, r7113;
}
{
sub.f16x2 r7119, r7101, r7116;
}
st.local.u32 [rd4+68], r7119;
{
add.f16x2 r7122, r5706, r5754;
}
{
mul.f16x2 r7125, r7122, r7052;
}
{
add.f16x2 r7128, r3953, r7125;
}
{
add.f16x2 r7131, r5722, r5738;
}
{
mul.f16x2 r7134, r7131, r7056;
}
{
add.f16x2 r7137, r7128, r7134;
}
{
sub.f16x2 r7140, r5712, r5760;
}
{
mul.f16x2 r7143, r7140, r7054;
}
{
sub.f16x2 r7146, r5728, r5744;
}
{
mul.f16x2 r7149, r7146, r7058;
}
{
add.f16x2 r7152, r7143, r7149;
}
{
add.f16x2 r7155, r7137, r7152;
}
st.local.u32 [rd4+188], r7155;
{
add.f16x2 r7158, r5706, r5754;
}
{
mul.f16x2 r7161, r7158, r7056;
}
{
add.f16x2 r7164, r3953, r7161;
}
{
add.f16x2 r7167, r5722, r5738;
}
{
mul.f16x2 r7170, r7167, r7060;
}
{
add.f16x2 r7173, r7164, r7170;
}
{
sub.f16x2 r7176, r5712, r5760;
}
{
mul.f16x2 r7179, r7176, r7058;
}
{
sub.f16x2 r7182, r5728, r5744;
}
{
mul.f16x2 r7185, r7182, r7061;
}
{
add.f16x2 r7188, r7179, r7185;
}
{
sub.f16x2 r7191, r7173, r7188;
}
st.local.u32 [rd4+108], r7191;
{
add.f16x2 r7194, r5706, r5754;
}
{
mul.f16x2 r7197, r7194, r7056;
}
{
add.f16x2 r7200, r3953, r7197;
}
{
add.f16x2 r7203, r5722, r5738;
}
{
mul.f16x2 r7206, r7203, r7060;
}
{
add.f16x2 r7209, r7200, r7206;
}
{
sub.f16x2 r7212, r5712, r5760;
}
{
mul.f16x2 r7215, r7212, r7058;
}
{
sub.f16x2 r7218, r5728, r5744;
}
{
mul.f16x2 r7221, r7218, r7061;
}
{
add.f16x2 r7224, r7215, r7221;
}
{
add.f16x2 r7227, r7209, r7224;
}
st.local.u32 [rd4+148], r7227;
{
add.f16x2 r7230, r5712, r5760;
}
{
mul.f16x2 r7233, r7230, r7052;
}
{
add.f16x2 r7236, r4097, r7233;
}
{
add.f16x2 r7239, r5728, r5744;
}
{
mul.f16x2 r7242, r7239, r7056;
}
{
add.f16x2 r7245, r7236, r7242;
}
{
sub.f16x2 r7248, r5706, r5754;
}
{
mul.f16x2 r7251, r7248, r7054;
}
{
sub.f16x2 r7254, r5722, r5738;
}
{
mul.f16x2 r7257, r7254, r7058;
}
{
add.f16x2 r7260, r7251, r7257;
}
{
add.f16x2 r7263, r7245, r7260;
}
st.local.u32 [rd4+72], r7263;
{
add.f16x2 r7266, r5712, r5760;
}
{
mul.f16x2 r7269, r7266, r7052;
}
{
add.f16x2 r7272, r4097, r7269;
}
{
add.f16x2 r7275, r5728, r5744;
}
{
mul.f16x2 r7278, r7275, r7056;
}
{
add.f16x2 r7281, r7272, r7278;
}
{
sub.f16x2 r7284, r5706, r5754;
}
{
mul.f16x2 r7287, r7284, r7054;
}
{
sub.f16x2 r7290, r5722, r5738;
}
{
mul.f16x2 r7293, r7290, r7058;
}
{
add.f16x2 r7296, r7287, r7293;
}
{
sub.f16x2 r7299, r7281, r7296;
}
st.local.u32 [rd4+192], r7299;
{
add.f16x2 r7302, r5712, r5760;
}
{
mul.f16x2 r7305, r7302, r7056;
}
{
add.f16x2 r7308, r4097, r7305;
}
{
add.f16x2 r7311, r5728, r5744;
}
{
mul.f16x2 r7314, r7311, r7060;
}
{
add.f16x2 r7317, r7308, r7314;
}
{
sub.f16x2 r7320, r5706, r5754;
}
{
mul.f16x2 r7323, r7320, r7058;
}
{
sub.f16x2 r7326, r5722, r5738;
}
{
mul.f16x2 r7329, r7326, r7061;
}
{
add.f16x2 r7332, r7323, r7329;
}
{
add.f16x2 r7335, r7317, r7332;
}
st.local.u32 [rd4+112], r7335;
{
add.f16x2 r7338, r5712, r5760;
}
{
mul.f16x2 r7341, r7338, r7056;
}
{
add.f16x2 r7344, r4097, r7341;
}
{
add.f16x2 r7347, r5728, r5744;
}
{
mul.f16x2 r7350, r7347, r7060;
}
{
add.f16x2 r7353, r7344, r7350;
}
{
sub.f16x2 r7356, r5706, r5754;
}
{
mul.f16x2 r7359, r7356, r7058;
}
{
sub.f16x2 r7362, r5722, r5738;
}
{
mul.f16x2 r7365, r7362, r7061;
}
{
add.f16x2 r7368, r7359, r7365;
}
{
sub.f16x2 r7371, r7353, r7368;
}
st.local.u32 [rd4+152], r7371;
mul.wide.u32 rd16, r3, 1374389535;
shr.u64 rd17, rd16, 35;
cvt.u32.u64 r15, rd17;
cvt.rn.f32.u32 f460, r15;
mul.f32 f461, f460, 0f3C24B5BE;
cos.approx.f32 f456, f461;
sin.approx.f32 f462, f461;
neg.f32 f457, f462;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f457;
mov.b32 r11413, {low, high};
}
mov.u32 r11411, -8;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11413;
mov.b32 r7502, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11413;
mov.b32 r7504, {high, high};
}
mov.u64 rd20, rd3;
bra.uni LBB0_4;
LBB0_5:
ld.local.u32 r11412, [rd20+72];
add.s64 rd20, rd20, 64;
LBB0_4:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11413;
mov.b32 r7482, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11413;
mov.b32 r7484, {high, high};
}
ld.local.u32 r7487, [rd20+12];
{
mul.f16x2 r7486, r7487, r7484;
}
{
fma.rn.f16x2 r7489, r11412, r7482, r7486;
}
st.local.u32 [rd20+8], r7489;
{
mul.f16x2 r7493, r11412, r7484;
}
{
neg.f16x2 r7496, r7493;
}
{
fma.rn.f16x2 r7498, r7487, r7482, r7496;
}
st.local.u32 [rd20+12], r7498;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7506, {low, high};
}
{
mul.f16x2 r7507, r7504, r7506;
}
{
mul.f16x2 r7510, r11413, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11413;
mov.b32 r7513, {high, low};
}
{
fma.rn.f16x2 r7515, r7507, r7513, r7510;
}
ld.local.u32 r7531, [rd20+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7515;
mov.b32 r7519, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7515;
mov.b32 r7521, {high, high};
}
ld.local.u32 r7536, [rd20+20];
{
mul.f16x2 r7523, r7536, r7521;
}
{
fma.rn.f16x2 r7526, r7531, r7519, r7523;
}
st.local.u32 [rd20+16], r7526;
{
mul.f16x2 r7530, r7531, r7521;
}
{
neg.f16x2 r7533, r7530;
}
{
fma.rn.f16x2 r7535, r7536, r7519, r7533;
}
st.local.u32 [rd20+20], r7535;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7543, {low, high};
}
{
mul.f16x2 r7544, r7504, r7543;
}
{
mul.f16x2 r7547, r7515, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7515;
mov.b32 r7550, {high, low};
}
{
fma.rn.f16x2 r7552, r7544, r7550, r7547;
}
ld.local.u32 r7568, [rd20+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7552;
mov.b32 r7556, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7552;
mov.b32 r7558, {high, high};
}
ld.local.u32 r7573, [rd20+28];
{
mul.f16x2 r7560, r7573, r7558;
}
{
fma.rn.f16x2 r7563, r7568, r7556, r7560;
}
st.local.u32 [rd20+24], r7563;
{
mul.f16x2 r7567, r7568, r7558;
}
{
neg.f16x2 r7570, r7567;
}
{
fma.rn.f16x2 r7572, r7573, r7556, r7570;
}
st.local.u32 [rd20+28], r7572;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7580, {low, high};
}
{
mul.f16x2 r7581, r7504, r7580;
}
{
mul.f16x2 r7584, r7552, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7552;
mov.b32 r7587, {high, low};
}
{
fma.rn.f16x2 r7589, r7581, r7587, r7584;
}
ld.local.u32 r7605, [rd20+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7589;
mov.b32 r7593, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7589;
mov.b32 r7595, {high, high};
}
ld.local.u32 r7610, [rd20+36];
{
mul.f16x2 r7597, r7610, r7595;
}
{
fma.rn.f16x2 r7600, r7605, r7593, r7597;
}
st.local.u32 [rd20+32], r7600;
{
mul.f16x2 r7604, r7605, r7595;
}
{
neg.f16x2 r7607, r7604;
}
{
fma.rn.f16x2 r7609, r7610, r7593, r7607;
}
st.local.u32 [rd20+36], r7609;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7617, {low, high};
}
{
mul.f16x2 r7618, r7504, r7617;
}
{
mul.f16x2 r7621, r7589, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7589;
mov.b32 r7624, {high, low};
}
{
fma.rn.f16x2 r7626, r7618, r7624, r7621;
}
ld.local.u32 r7638, [rd20+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7626;
mov.b32 r7630, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7626;
mov.b32 r7632, {high, high};
}
ld.local.u32 r7647, [rd20+44];
{
mul.f16x2 r7634, r7647, r7632;
}
{
fma.rn.f16x2 r7637, r7638, r7630, r7634;
}
st.local.u32 [rd20+40], r7637;
{
mul.f16x2 r7641, r7638, r7632;
}
{
neg.f16x2 r7644, r7641;
}
{
fma.rn.f16x2 r7646, r7647, r7630, r7644;
}
st.local.u32 [rd20+44], r7646;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7654, {low, high};
}
{
mul.f16x2 r7655, r7504, r7654;
}
{
mul.f16x2 r7658, r7626, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7626;
mov.b32 r7661, {high, low};
}
{
fma.rn.f16x2 r7663, r7655, r7661, r7658;
}
ld.local.u32 r7675, [rd20+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7663;
mov.b32 r7667, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7663;
mov.b32 r7669, {high, high};
}
ld.local.u32 r7672, [rd20+52];
{
mul.f16x2 r7671, r7672, r7669;
}
{
fma.rn.f16x2 r7674, r7675, r7667, r7671;
}
st.local.u32 [rd20+48], r7674;
{
mul.f16x2 r7678, r7675, r7669;
}
{
neg.f16x2 r7681, r7678;
}
{
fma.rn.f16x2 r7683, r7672, r7667, r7681;
}
st.local.u32 [rd20+52], r7683;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7691, {low, high};
}
{
mul.f16x2 r7692, r7504, r7691;
}
{
mul.f16x2 r7695, r7663, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7663;
mov.b32 r7698, {high, low};
}
{
fma.rn.f16x2 r7700, r7692, r7698, r7695;
}
ld.local.u32 r7712, [rd20+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7700;
mov.b32 r7704, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7700;
mov.b32 r7706, {high, high};
}
ld.local.u32 r7709, [rd20+60];
{
mul.f16x2 r7708, r7709, r7706;
}
{
fma.rn.f16x2 r7711, r7712, r7704, r7708;
}
st.local.u32 [rd20+56], r7711;
{
mul.f16x2 r7715, r7712, r7706;
}
{
neg.f16x2 r7718, r7715;
}
{
fma.rn.f16x2 r7720, r7709, r7704, r7718;
}
st.local.u32 [rd20+60], r7720;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7728, {low, high};
}
{
mul.f16x2 r7729, r7504, r7728;
}
{
mul.f16x2 r7732, r7700, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7700;
mov.b32 r7735, {high, low};
}
{
fma.rn.f16x2 r7737, r7729, r7735, r7732;
}
ld.local.u32 r7749, [rd20+64];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7737;
mov.b32 r7741, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7737;
mov.b32 r7743, {high, high};
}
ld.local.u32 r7746, [rd20+68];
{
mul.f16x2 r7745, r7746, r7743;
}
{
fma.rn.f16x2 r7748, r7749, r7741, r7745;
}
st.local.u32 [rd20+64], r7748;
{
mul.f16x2 r7752, r7749, r7743;
}
{
neg.f16x2 r7755, r7752;
}
{
fma.rn.f16x2 r7757, r7746, r7741, r7755;
}
st.local.u32 [rd20+68], r7757;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7765, {low, high};
}
{
mul.f16x2 r7766, r7504, r7765;
}
{
mul.f16x2 r7769, r7737, r7502;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7737;
mov.b32 r7772, {high, low};
}
{
fma.rn.f16x2 r11413, r7766, r7772, r7769;
}
add.s32 r11411, r11411, 8;
setp.eq.s32 p2, r11411, 16;
@p2 bra LBB0_6;
bra.uni LBB0_5;
LBB0_6:
mul.lo.s32 r11303, r15, 25;
sub.s32 r11304, r3, r11303;
shl.b32 r11305, r11304, 3;
add.s32 r11306, r12, r11305;
barrier.sync 0;
mad.lo.s32 r11307, r15, 5000, r11306;
ld.local.v2.u32 {r11308, r11309}, [rd3];
st.shared.u32 [r11307], r11308;
st.shared.u32 [r11307+4], r11309;
ld.local.v2.u32 {r11312, r11313}, [rd4+4];
st.shared.u32 [r11307+200], r11312;
st.shared.u32 [r11307+204], r11313;
ld.local.v2.u32 {r11316, r11317}, [rd4+12];
st.shared.u32 [r11307+400], r11316;
st.shared.u32 [r11307+404], r11317;
ld.local.v2.u32 {r11320, r11321}, [rd4+20];
st.shared.u32 [r11307+600], r11320;
st.shared.u32 [r11307+604], r11321;
ld.local.v2.u32 {r11324, r11325}, [rd4+28];
st.shared.u32 [r11307+800], r11324;
st.shared.u32 [r11307+804], r11325;
ld.local.v2.u32 {r11328, r11329}, [rd4+36];
st.shared.u32 [r11307+1000], r11328;
st.shared.u32 [r11307+1004], r11329;
ld.local.v2.u32 {r11332, r11333}, [rd4+44];
st.shared.u32 [r11307+1200], r11332;
st.shared.u32 [r11307+1204], r11333;
ld.local.v2.u32 {r11336, r11337}, [rd4+52];
st.shared.u32 [r11307+1400], r11336;
st.shared.u32 [r11307+1404], r11337;
ld.local.v2.u32 {r11340, r11341}, [rd4+60];
st.shared.u32 [r11307+1600], r11340;
st.shared.u32 [r11307+1604], r11341;
ld.local.v2.u32 {r11344, r11345}, [rd4+68];
st.shared.u32 [r11307+1800], r11344;
st.shared.u32 [r11307+1804], r11345;
ld.local.v2.u32 {r11348, r11349}, [rd4+76];
st.shared.u32 [r11307+2000], r11348;
st.shared.u32 [r11307+2004], r11349;
ld.local.v2.u32 {r11352, r11353}, [rd4+84];
st.shared.u32 [r11307+2200], r11352;
st.shared.u32 [r11307+2204], r11353;
ld.local.v2.u32 {r11356, r11357}, [rd4+92];
st.shared.u32 [r11307+2400], r11356;
st.shared.u32 [r11307+2404], r11357;
ld.local.v2.u32 {r11360, r11361}, [rd4+100];
st.shared.u32 [r11307+2600], r11360;
st.shared.u32 [r11307+2604], r11361;
ld.local.v2.u32 {r11364, r11365}, [rd4+108];
st.shared.u32 [r11307+2800], r11364;
st.shared.u32 [r11307+2804], r11365;
ld.local.v2.u32 {r11368, r11369}, [rd4+116];
st.shared.u32 [r11307+3000], r11368;
st.shared.u32 [r11307+3004], r11369;
ld.local.v2.u32 {r11372, r11373}, [rd4+124];
st.shared.u32 [r11307+3200], r11372;
st.shared.u32 [r11307+3204], r11373;
ld.local.v2.u32 {r11376, r11377}, [rd4+132];
st.shared.u32 [r11307+3400], r11376;
st.shared.u32 [r11307+3404], r11377;
ld.local.v2.u32 {r11380, r11381}, [rd4+140];
st.shared.u32 [r11307+3600], r11380;
st.shared.u32 [r11307+3604], r11381;
ld.local.v2.u32 {r11384, r11385}, [rd4+148];
st.shared.u32 [r11307+3800], r11384;
st.shared.u32 [r11307+3804], r11385;
ld.local.v2.u32 {r11388, r11389}, [rd4+156];
st.shared.u32 [r11307+4000], r11388;
st.shared.u32 [r11307+4004], r11389;
ld.local.v2.u32 {r11392, r11393}, [rd4+164];
st.shared.u32 [r11307+4200], r11392;
st.shared.u32 [r11307+4204], r11393;
ld.local.v2.u32 {r11396, r11397}, [rd4+172];
st.shared.u32 [r11307+4400], r11396;
st.shared.u32 [r11307+4404], r11397;
ld.local.v2.u32 {r11400, r11401}, [rd4+180];
st.shared.u32 [r11307+4600], r11400;
st.shared.u32 [r11307+4604], r11401;
ld.local.v2.u32 {r11404, r11405}, [rd4+188];
st.shared.u32 [r11307+4800], r11404;
st.shared.u32 [r11307+4804], r11405;
barrier.sync 0;
ld.shared.u32 r7792, [r13];
ld.shared.u32 r7804, [r13+4];
ld.shared.u32 r8114, [r13+5000];
ld.shared.u32 r8126, [r13+5004];
ld.shared.u32 r8436, [r13+10000];
ld.shared.u32 r8448, [r13+10004];
ld.shared.u32 r8758, [r13+15000];
ld.shared.u32 r8770, [r13+15004];
ld.shared.u32 r9080, [r13+20000];
ld.shared.u32 r9092, [r13+20004];
ld.shared.u32 r7789, [r13+25000];
ld.shared.u32 r7801, [r13+25004];
ld.shared.u32 r8111, [r13+30000];
ld.shared.u32 r8123, [r13+30004];
ld.shared.u32 r8433, [r13+35000];
ld.shared.u32 r8445, [r13+35004];
ld.shared.u32 r8755, [r13+40000];
ld.shared.u32 r8767, [r13+40004];
ld.shared.u32 r9077, [r13+45000];
ld.shared.u32 r9089, [r13+45004];
ld.shared.u32 r7795, [r13+50000];
ld.shared.u32 r7807, [r13+50004];
ld.shared.u32 r8117, [r13+55000];
ld.shared.u32 r8129, [r13+55004];
ld.shared.u32 r8439, [r13+60000];
ld.shared.u32 r8451, [r13+60004];
ld.shared.u32 r8761, [r13+65000];
ld.shared.u32 r8773, [r13+65004];
ld.shared.u32 r9083, [r13+70000];
ld.shared.u32 r9095, [r13+70004];
ld.shared.u32 r7796, [r13+75000];
ld.shared.u32 r7808, [r13+75004];
ld.shared.u32 r8118, [r13+80000];
ld.shared.u32 r8130, [r13+80004];
ld.shared.u32 r8440, [r13+85000];
ld.shared.u32 r8452, [r13+85004];
ld.shared.u32 r8762, [r13+90000];
ld.shared.u32 r8774, [r13+90004];
ld.shared.u32 r9084, [r13+95000];
ld.shared.u32 r9096, [r13+95004];
ld.shared.u32 r7790, [r13+100000];
ld.shared.u32 r7802, [r13+100004];
ld.shared.u32 r8112, [r13+105000];
ld.shared.u32 r8124, [r13+105004];
ld.shared.u32 r8434, [r13+110000];
ld.shared.u32 r8446, [r13+110004];
ld.shared.u32 r8756, [r13+115000];
ld.shared.u32 r8768, [r13+115004];
ld.shared.u32 r9078, [r13+120000];
ld.shared.u32 r9090, [r13+120004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7778, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7779, {low, high};
}
{
neg.f16x2 r7780, r7779;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7782, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7783, {low, high};
}
{
neg.f16x2 r7784, r7783;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7786, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7787, {low, high};
}
{
add.f16x2 r7788, r7789, r7790;
}
{
add.f16x2 r7791, r7792, r7788;
}
{
add.f16x2 r7794, r7795, r7796;
}
{
add.f16x2 r7797, r7791, r7794;
}
{
add.f16x2 r7800, r7801, r7802;
}
{
add.f16x2 r7803, r7804, r7800;
}
{
add.f16x2 r7806, r7807, r7808;
}
{
add.f16x2 r7809, r7803, r7806;
}
{
add.f16x2 r7812, r7789, r7790;
}
{
mul.f16x2 r7815, r7812, r7778;
}
{
add.f16x2 r7818, r7792, r7815;
}
{
add.f16x2 r7821, r7795, r7796;
}
{
mul.f16x2 r7824, r7821, r7782;
}
{
add.f16x2 r7827, r7818, r7824;
}
{
sub.f16x2 r7830, r7801, r7802;
}
{
mul.f16x2 r7833, r7830, r7780;
}
{
sub.f16x2 r7836, r7807, r7808;
}
{
mul.f16x2 r7839, r7836, r7784;
}
{
add.f16x2 r7842, r7833, r7839;
}
{
sub.f16x2 r7845, r7827, r7842;
}
{
add.f16x2 r7848, r7789, r7790;
}
{
mul.f16x2 r7851, r7848, r7778;
}
{
add.f16x2 r7854, r7792, r7851;
}
{
add.f16x2 r7857, r7795, r7796;
}
{
mul.f16x2 r7860, r7857, r7782;
}
{
add.f16x2 r7863, r7854, r7860;
}
{
sub.f16x2 r7866, r7801, r7802;
}
{
mul.f16x2 r7869, r7866, r7780;
}
{
sub.f16x2 r7872, r7807, r7808;
}
{
mul.f16x2 r7875, r7872, r7784;
}
{
add.f16x2 r7878, r7869, r7875;
}
{
add.f16x2 r7881, r7863, r7878;
}
{
add.f16x2 r7884, r7789, r7790;
}
{
mul.f16x2 r7887, r7884, r7782;
}
{
add.f16x2 r7890, r7792, r7887;
}
{
add.f16x2 r7893, r7795, r7796;
}
{
mul.f16x2 r7896, r7893, r7786;
}
{
add.f16x2 r7899, r7890, r7896;
}
{
sub.f16x2 r7902, r7801, r7802;
}
{
mul.f16x2 r7905, r7902, r7784;
}
{
sub.f16x2 r7908, r7807, r7808;
}
{
mul.f16x2 r7911, r7908, r7787;
}
{
add.f16x2 r7914, r7905, r7911;
}
{
sub.f16x2 r7917, r7899, r7914;
}
{
add.f16x2 r7920, r7789, r7790;
}
{
mul.f16x2 r7923, r7920, r7782;
}
{
add.f16x2 r7926, r7792, r7923;
}
{
add.f16x2 r7929, r7795, r7796;
}
{
mul.f16x2 r7932, r7929, r7786;
}
{
add.f16x2 r7935, r7926, r7932;
}
{
sub.f16x2 r7938, r7801, r7802;
}
{
mul.f16x2 r7941, r7938, r7784;
}
{
sub.f16x2 r7944, r7807, r7808;
}
{
mul.f16x2 r7947, r7944, r7787;
}
{
add.f16x2 r7950, r7941, r7947;
}
{
add.f16x2 r7953, r7935, r7950;
}
{
add.f16x2 r7956, r7801, r7802;
}
{
mul.f16x2 r7959, r7956, r7778;
}
{
add.f16x2 r7962, r7804, r7959;
}
{
add.f16x2 r7965, r7807, r7808;
}
{
mul.f16x2 r7968, r7965, r7782;
}
{
add.f16x2 r7971, r7962, r7968;
}
{
sub.f16x2 r7974, r7789, r7790;
}
{
mul.f16x2 r7977, r7974, r7780;
}
{
sub.f16x2 r7980, r7795, r7796;
}
{
mul.f16x2 r7983, r7980, r7784;
}
{
add.f16x2 r7986, r7977, r7983;
}
{
add.f16x2 r7989, r7971, r7986;
}
{
add.f16x2 r7992, r7801, r7802;
}
{
mul.f16x2 r7995, r7992, r7778;
}
{
add.f16x2 r7998, r7804, r7995;
}
{
add.f16x2 r8001, r7807, r7808;
}
{
mul.f16x2 r8004, r8001, r7782;
}
{
add.f16x2 r8007, r7998, r8004;
}
{
sub.f16x2 r8010, r7789, r7790;
}
{
mul.f16x2 r8013, r8010, r7780;
}
{
sub.f16x2 r8016, r7795, r7796;
}
{
mul.f16x2 r8019, r8016, r7784;
}
{
add.f16x2 r8022, r8013, r8019;
}
{
sub.f16x2 r8025, r8007, r8022;
}
{
add.f16x2 r8028, r7801, r7802;
}
{
mul.f16x2 r8031, r8028, r7782;
}
{
add.f16x2 r8034, r7804, r8031;
}
{
add.f16x2 r8037, r7807, r7808;
}
{
mul.f16x2 r8040, r8037, r7786;
}
{
add.f16x2 r8043, r8034, r8040;
}
{
sub.f16x2 r8046, r7789, r7790;
}
{
mul.f16x2 r8049, r8046, r7784;
}
{
sub.f16x2 r8052, r7795, r7796;
}
{
mul.f16x2 r8055, r8052, r7787;
}
{
add.f16x2 r8058, r8049, r8055;
}
{
add.f16x2 r8061, r8043, r8058;
}
{
add.f16x2 r8064, r7801, r7802;
}
{
mul.f16x2 r8067, r8064, r7782;
}
{
add.f16x2 r8070, r7804, r8067;
}
{
add.f16x2 r8073, r7807, r7808;
}
{
mul.f16x2 r8076, r8073, r7786;
}
{
add.f16x2 r8079, r8070, r8076;
}
{
sub.f16x2 r8082, r7789, r7790;
}
{
mul.f16x2 r8085, r8082, r7784;
}
{
sub.f16x2 r8088, r7795, r7796;
}
{
mul.f16x2 r8091, r8088, r7787;
}
{
add.f16x2 r8094, r8085, r8091;
}
{
sub.f16x2 r8097, r8079, r8094;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8100, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8101, {low, high};
}
{
neg.f16x2 r8102, r8101;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8104, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8105, {low, high};
}
{
neg.f16x2 r8106, r8105;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8108, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8109, {low, high};
}
{
add.f16x2 r8110, r8111, r8112;
}
{
add.f16x2 r8113, r8114, r8110;
}
{
add.f16x2 r8116, r8117, r8118;
}
{
add.f16x2 r8119, r8113, r8116;
}
{
add.f16x2 r8122, r8123, r8124;
}
{
add.f16x2 r8125, r8126, r8122;
}
{
add.f16x2 r8128, r8129, r8130;
}
{
add.f16x2 r8131, r8125, r8128;
}
{
add.f16x2 r8134, r8111, r8112;
}
{
mul.f16x2 r8137, r8134, r8100;
}
{
add.f16x2 r8140, r8114, r8137;
}
{
add.f16x2 r8143, r8117, r8118;
}
{
mul.f16x2 r8146, r8143, r8104;
}
{
add.f16x2 r8149, r8140, r8146;
}
{
sub.f16x2 r8152, r8123, r8124;
}
{
mul.f16x2 r8155, r8152, r8102;
}
{
sub.f16x2 r8158, r8129, r8130;
}
{
mul.f16x2 r8161, r8158, r8106;
}
{
add.f16x2 r8164, r8155, r8161;
}
{
sub.f16x2 r8167, r8149, r8164;
}
{
add.f16x2 r8170, r8111, r8112;
}
{
mul.f16x2 r8173, r8170, r8100;
}
{
add.f16x2 r8176, r8114, r8173;
}
{
add.f16x2 r8179, r8117, r8118;
}
{
mul.f16x2 r8182, r8179, r8104;
}
{
add.f16x2 r8185, r8176, r8182;
}
{
sub.f16x2 r8188, r8123, r8124;
}
{
mul.f16x2 r8191, r8188, r8102;
}
{
sub.f16x2 r8194, r8129, r8130;
}
{
mul.f16x2 r8197, r8194, r8106;
}
{
add.f16x2 r8200, r8191, r8197;
}
{
add.f16x2 r8203, r8185, r8200;
}
{
add.f16x2 r8206, r8111, r8112;
}
{
mul.f16x2 r8209, r8206, r8104;
}
{
add.f16x2 r8212, r8114, r8209;
}
{
add.f16x2 r8215, r8117, r8118;
}
{
mul.f16x2 r8218, r8215, r8108;
}
{
add.f16x2 r8221, r8212, r8218;
}
{
sub.f16x2 r8224, r8123, r8124;
}
{
mul.f16x2 r8227, r8224, r8106;
}
{
sub.f16x2 r8230, r8129, r8130;
}
{
mul.f16x2 r8233, r8230, r8109;
}
{
add.f16x2 r8236, r8227, r8233;
}
{
sub.f16x2 r8239, r8221, r8236;
}
{
add.f16x2 r8242, r8111, r8112;
}
{
mul.f16x2 r8245, r8242, r8104;
}
{
add.f16x2 r8248, r8114, r8245;
}
{
add.f16x2 r8251, r8117, r8118;
}
{
mul.f16x2 r8254, r8251, r8108;
}
{
add.f16x2 r8257, r8248, r8254;
}
{
sub.f16x2 r8260, r8123, r8124;
}
{
mul.f16x2 r8263, r8260, r8106;
}
{
sub.f16x2 r8266, r8129, r8130;
}
{
mul.f16x2 r8269, r8266, r8109;
}
{
add.f16x2 r8272, r8263, r8269;
}
{
add.f16x2 r8275, r8257, r8272;
}
{
add.f16x2 r8278, r8123, r8124;
}
{
mul.f16x2 r8281, r8278, r8100;
}
{
add.f16x2 r8284, r8126, r8281;
}
{
add.f16x2 r8287, r8129, r8130;
}
{
mul.f16x2 r8290, r8287, r8104;
}
{
add.f16x2 r8293, r8284, r8290;
}
{
sub.f16x2 r8296, r8111, r8112;
}
{
mul.f16x2 r8299, r8296, r8102;
}
{
sub.f16x2 r8302, r8117, r8118;
}
{
mul.f16x2 r8305, r8302, r8106;
}
{
add.f16x2 r8308, r8299, r8305;
}
{
add.f16x2 r8311, r8293, r8308;
}
{
add.f16x2 r8314, r8123, r8124;
}
{
mul.f16x2 r8317, r8314, r8100;
}
{
add.f16x2 r8320, r8126, r8317;
}
{
add.f16x2 r8323, r8129, r8130;
}
{
mul.f16x2 r8326, r8323, r8104;
}
{
add.f16x2 r8329, r8320, r8326;
}
{
sub.f16x2 r8332, r8111, r8112;
}
{
mul.f16x2 r8335, r8332, r8102;
}
{
sub.f16x2 r8338, r8117, r8118;
}
{
mul.f16x2 r8341, r8338, r8106;
}
{
add.f16x2 r8344, r8335, r8341;
}
{
sub.f16x2 r8347, r8329, r8344;
}
{
add.f16x2 r8350, r8123, r8124;
}
{
mul.f16x2 r8353, r8350, r8104;
}
{
add.f16x2 r8356, r8126, r8353;
}
{
add.f16x2 r8359, r8129, r8130;
}
{
mul.f16x2 r8362, r8359, r8108;
}
{
add.f16x2 r8365, r8356, r8362;
}
{
sub.f16x2 r8368, r8111, r8112;
}
{
mul.f16x2 r8371, r8368, r8106;
}
{
sub.f16x2 r8374, r8117, r8118;
}
{
mul.f16x2 r8377, r8374, r8109;
}
{
add.f16x2 r8380, r8371, r8377;
}
{
add.f16x2 r8383, r8365, r8380;
}
{
add.f16x2 r8386, r8123, r8124;
}
{
mul.f16x2 r8389, r8386, r8104;
}
{
add.f16x2 r8392, r8126, r8389;
}
{
add.f16x2 r8395, r8129, r8130;
}
{
mul.f16x2 r8398, r8395, r8108;
}
{
add.f16x2 r8401, r8392, r8398;
}
{
sub.f16x2 r8404, r8111, r8112;
}
{
mul.f16x2 r8407, r8404, r8106;
}
{
sub.f16x2 r8410, r8117, r8118;
}
{
mul.f16x2 r8413, r8410, r8109;
}
{
add.f16x2 r8416, r8407, r8413;
}
{
sub.f16x2 r8419, r8401, r8416;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8422, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8423, {low, high};
}
{
neg.f16x2 r8424, r8423;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8426, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8427, {low, high};
}
{
neg.f16x2 r8428, r8427;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8430, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8431, {low, high};
}
{
add.f16x2 r8432, r8433, r8434;
}
{
add.f16x2 r8435, r8436, r8432;
}
{
add.f16x2 r8438, r8439, r8440;
}
{
add.f16x2 r8441, r8435, r8438;
}
{
add.f16x2 r8444, r8445, r8446;
}
{
add.f16x2 r8447, r8448, r8444;
}
{
add.f16x2 r8450, r8451, r8452;
}
{
add.f16x2 r8453, r8447, r8450;
}
{
add.f16x2 r8456, r8433, r8434;
}
{
mul.f16x2 r8459, r8456, r8422;
}
{
add.f16x2 r8462, r8436, r8459;
}
{
add.f16x2 r8465, r8439, r8440;
}
{
mul.f16x2 r8468, r8465, r8426;
}
{
add.f16x2 r8471, r8462, r8468;
}
{
sub.f16x2 r8474, r8445, r8446;
}
{
mul.f16x2 r8477, r8474, r8424;
}
{
sub.f16x2 r8480, r8451, r8452;
}
{
mul.f16x2 r8483, r8480, r8428;
}
{
add.f16x2 r8486, r8477, r8483;
}
{
sub.f16x2 r8489, r8471, r8486;
}
{
add.f16x2 r8492, r8433, r8434;
}
{
mul.f16x2 r8495, r8492, r8422;
}
{
add.f16x2 r8498, r8436, r8495;
}
{
add.f16x2 r8501, r8439, r8440;
}
{
mul.f16x2 r8504, r8501, r8426;
}
{
add.f16x2 r8507, r8498, r8504;
}
{
sub.f16x2 r8510, r8445, r8446;
}
{
mul.f16x2 r8513, r8510, r8424;
}
{
sub.f16x2 r8516, r8451, r8452;
}
{
mul.f16x2 r8519, r8516, r8428;
}
{
add.f16x2 r8522, r8513, r8519;
}
{
add.f16x2 r8525, r8507, r8522;
}
{
add.f16x2 r8528, r8433, r8434;
}
{
mul.f16x2 r8531, r8528, r8426;
}
{
add.f16x2 r8534, r8436, r8531;
}
{
add.f16x2 r8537, r8439, r8440;
}
{
mul.f16x2 r8540, r8537, r8430;
}
{
add.f16x2 r8543, r8534, r8540;
}
{
sub.f16x2 r8546, r8445, r8446;
}
{
mul.f16x2 r8549, r8546, r8428;
}
{
sub.f16x2 r8552, r8451, r8452;
}
{
mul.f16x2 r8555, r8552, r8431;
}
{
add.f16x2 r8558, r8549, r8555;
}
{
sub.f16x2 r8561, r8543, r8558;
}
{
add.f16x2 r8564, r8433, r8434;
}
{
mul.f16x2 r8567, r8564, r8426;
}
{
add.f16x2 r8570, r8436, r8567;
}
{
add.f16x2 r8573, r8439, r8440;
}
{
mul.f16x2 r8576, r8573, r8430;
}
{
add.f16x2 r8579, r8570, r8576;
}
{
sub.f16x2 r8582, r8445, r8446;
}
{
mul.f16x2 r8585, r8582, r8428;
}
{
sub.f16x2 r8588, r8451, r8452;
}
{
mul.f16x2 r8591, r8588, r8431;
}
{
add.f16x2 r8594, r8585, r8591;
}
{
add.f16x2 r8597, r8579, r8594;
}
{
add.f16x2 r8600, r8445, r8446;
}
{
mul.f16x2 r8603, r8600, r8422;
}
{
add.f16x2 r8606, r8448, r8603;
}
{
add.f16x2 r8609, r8451, r8452;
}
{
mul.f16x2 r8612, r8609, r8426;
}
{
add.f16x2 r8615, r8606, r8612;
}
{
sub.f16x2 r8618, r8433, r8434;
}
{
mul.f16x2 r8621, r8618, r8424;
}
{
sub.f16x2 r8624, r8439, r8440;
}
{
mul.f16x2 r8627, r8624, r8428;
}
{
add.f16x2 r8630, r8621, r8627;
}
{
add.f16x2 r8633, r8615, r8630;
}
{
add.f16x2 r8636, r8445, r8446;
}
{
mul.f16x2 r8639, r8636, r8422;
}
{
add.f16x2 r8642, r8448, r8639;
}
{
add.f16x2 r8645, r8451, r8452;
}
{
mul.f16x2 r8648, r8645, r8426;
}
{
add.f16x2 r8651, r8642, r8648;
}
{
sub.f16x2 r8654, r8433, r8434;
}
{
mul.f16x2 r8657, r8654, r8424;
}
{
sub.f16x2 r8660, r8439, r8440;
}
{
mul.f16x2 r8663, r8660, r8428;
}
{
add.f16x2 r8666, r8657, r8663;
}
{
sub.f16x2 r8669, r8651, r8666;
}
{
add.f16x2 r8672, r8445, r8446;
}
{
mul.f16x2 r8675, r8672, r8426;
}
{
add.f16x2 r8678, r8448, r8675;
}
{
add.f16x2 r8681, r8451, r8452;
}
{
mul.f16x2 r8684, r8681, r8430;
}
{
add.f16x2 r8687, r8678, r8684;
}
{
sub.f16x2 r8690, r8433, r8434;
}
{
mul.f16x2 r8693, r8690, r8428;
}
{
sub.f16x2 r8696, r8439, r8440;
}
{
mul.f16x2 r8699, r8696, r8431;
}
{
add.f16x2 r8702, r8693, r8699;
}
{
add.f16x2 r8705, r8687, r8702;
}
{
add.f16x2 r8708, r8445, r8446;
}
{
mul.f16x2 r8711, r8708, r8426;
}
{
add.f16x2 r8714, r8448, r8711;
}
{
add.f16x2 r8717, r8451, r8452;
}
{
mul.f16x2 r8720, r8717, r8430;
}
{
add.f16x2 r8723, r8714, r8720;
}
{
sub.f16x2 r8726, r8433, r8434;
}
{
mul.f16x2 r8729, r8726, r8428;
}
{
sub.f16x2 r8732, r8439, r8440;
}
{
mul.f16x2 r8735, r8732, r8431;
}
{
add.f16x2 r8738, r8729, r8735;
}
{
sub.f16x2 r8741, r8723, r8738;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8744, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8745, {low, high};
}
{
neg.f16x2 r8746, r8745;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8748, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8749, {low, high};
}
{
neg.f16x2 r8750, r8749;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8752, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8753, {low, high};
}
{
add.f16x2 r8754, r8755, r8756;
}
{
add.f16x2 r8757, r8758, r8754;
}
{
add.f16x2 r8760, r8761, r8762;
}
{
add.f16x2 r8763, r8757, r8760;
}
{
add.f16x2 r8766, r8767, r8768;
}
{
add.f16x2 r8769, r8770, r8766;
}
{
add.f16x2 r8772, r8773, r8774;
}
{
add.f16x2 r8775, r8769, r8772;
}
{
add.f16x2 r8778, r8755, r8756;
}
{
mul.f16x2 r8781, r8778, r8744;
}
{
add.f16x2 r8784, r8758, r8781;
}
{
add.f16x2 r8787, r8761, r8762;
}
{
mul.f16x2 r8790, r8787, r8748;
}
{
add.f16x2 r8793, r8784, r8790;
}
{
sub.f16x2 r8796, r8767, r8768;
}
{
mul.f16x2 r8799, r8796, r8746;
}
{
sub.f16x2 r8802, r8773, r8774;
}
{
mul.f16x2 r8805, r8802, r8750;
}
{
add.f16x2 r8808, r8799, r8805;
}
{
sub.f16x2 r8811, r8793, r8808;
}
{
add.f16x2 r8814, r8755, r8756;
}
{
mul.f16x2 r8817, r8814, r8744;
}
{
add.f16x2 r8820, r8758, r8817;
}
{
add.f16x2 r8823, r8761, r8762;
}
{
mul.f16x2 r8826, r8823, r8748;
}
{
add.f16x2 r8829, r8820, r8826;
}
{
sub.f16x2 r8832, r8767, r8768;
}
{
mul.f16x2 r8835, r8832, r8746;
}
{
sub.f16x2 r8838, r8773, r8774;
}
{
mul.f16x2 r8841, r8838, r8750;
}
{
add.f16x2 r8844, r8835, r8841;
}
{
add.f16x2 r8847, r8829, r8844;
}
{
add.f16x2 r8850, r8755, r8756;
}
{
mul.f16x2 r8853, r8850, r8748;
}
{
add.f16x2 r8856, r8758, r8853;
}
{
add.f16x2 r8859, r8761, r8762;
}
{
mul.f16x2 r8862, r8859, r8752;
}
{
add.f16x2 r8865, r8856, r8862;
}
{
sub.f16x2 r8868, r8767, r8768;
}
{
mul.f16x2 r8871, r8868, r8750;
}
{
sub.f16x2 r8874, r8773, r8774;
}
{
mul.f16x2 r8877, r8874, r8753;
}
{
add.f16x2 r8880, r8871, r8877;
}
{
sub.f16x2 r8883, r8865, r8880;
}
{
add.f16x2 r8886, r8755, r8756;
}
{
mul.f16x2 r8889, r8886, r8748;
}
{
add.f16x2 r8892, r8758, r8889;
}
{
add.f16x2 r8895, r8761, r8762;
}
{
mul.f16x2 r8898, r8895, r8752;
}
{
add.f16x2 r8901, r8892, r8898;
}
{
sub.f16x2 r8904, r8767, r8768;
}
{
mul.f16x2 r8907, r8904, r8750;
}
{
sub.f16x2 r8910, r8773, r8774;
}
{
mul.f16x2 r8913, r8910, r8753;
}
{
add.f16x2 r8916, r8907, r8913;
}
{
add.f16x2 r8919, r8901, r8916;
}
{
add.f16x2 r8922, r8767, r8768;
}
{
mul.f16x2 r8925, r8922, r8744;
}
{
add.f16x2 r8928, r8770, r8925;
}
{
add.f16x2 r8931, r8773, r8774;
}
{
mul.f16x2 r8934, r8931, r8748;
}
{
add.f16x2 r8937, r8928, r8934;
}
{
sub.f16x2 r8940, r8755, r8756;
}
{
mul.f16x2 r8943, r8940, r8746;
}
{
sub.f16x2 r8946, r8761, r8762;
}
{
mul.f16x2 r8949, r8946, r8750;
}
{
add.f16x2 r8952, r8943, r8949;
}
{
add.f16x2 r8955, r8937, r8952;
}
{
add.f16x2 r8958, r8767, r8768;
}
{
mul.f16x2 r8961, r8958, r8744;
}
{
add.f16x2 r8964, r8770, r8961;
}
{
add.f16x2 r8967, r8773, r8774;
}
{
mul.f16x2 r8970, r8967, r8748;
}
{
add.f16x2 r8973, r8964, r8970;
}
{
sub.f16x2 r8976, r8755, r8756;
}
{
mul.f16x2 r8979, r8976, r8746;
}
{
sub.f16x2 r8982, r8761, r8762;
}
{
mul.f16x2 r8985, r8982, r8750;
}
{
add.f16x2 r8988, r8979, r8985;
}
{
sub.f16x2 r8991, r8973, r8988;
}
{
add.f16x2 r8994, r8767, r8768;
}
{
mul.f16x2 r8997, r8994, r8748;
}
{
add.f16x2 r9000, r8770, r8997;
}
{
add.f16x2 r9003, r8773, r8774;
}
{
mul.f16x2 r9006, r9003, r8752;
}
{
add.f16x2 r9009, r9000, r9006;
}
{
sub.f16x2 r9012, r8755, r8756;
}
{
mul.f16x2 r9015, r9012, r8750;
}
{
sub.f16x2 r9018, r8761, r8762;
}
{
mul.f16x2 r9021, r9018, r8753;
}
{
add.f16x2 r9024, r9015, r9021;
}
{
add.f16x2 r9027, r9009, r9024;
}
{
add.f16x2 r9030, r8767, r8768;
}
{
mul.f16x2 r9033, r9030, r8748;
}
{
add.f16x2 r9036, r8770, r9033;
}
{
add.f16x2 r9039, r8773, r8774;
}
{
mul.f16x2 r9042, r9039, r8752;
}
{
add.f16x2 r9045, r9036, r9042;
}
{
sub.f16x2 r9048, r8755, r8756;
}
{
mul.f16x2 r9051, r9048, r8750;
}
{
sub.f16x2 r9054, r8761, r8762;
}
{
mul.f16x2 r9057, r9054, r8753;
}
{
add.f16x2 r9060, r9051, r9057;
}
{
sub.f16x2 r9063, r9045, r9060;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9066, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9067, {low, high};
}
{
neg.f16x2 r9068, r9067;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9070, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9071, {low, high};
}
{
neg.f16x2 r9072, r9071;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9074, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9075, {low, high};
}
{
add.f16x2 r9076, r9077, r9078;
}
{
add.f16x2 r9079, r9080, r9076;
}
{
add.f16x2 r9082, r9083, r9084;
}
{
add.f16x2 r9085, r9079, r9082;
}
{
add.f16x2 r9088, r9089, r9090;
}
{
add.f16x2 r9091, r9092, r9088;
}
{
add.f16x2 r9094, r9095, r9096;
}
{
add.f16x2 r9097, r9091, r9094;
}
{
add.f16x2 r9100, r9077, r9078;
}
{
mul.f16x2 r9103, r9100, r9066;
}
{
add.f16x2 r9106, r9080, r9103;
}
{
add.f16x2 r9109, r9083, r9084;
}
{
mul.f16x2 r9112, r9109, r9070;
}
{
add.f16x2 r9115, r9106, r9112;
}
{
sub.f16x2 r9118, r9089, r9090;
}
{
mul.f16x2 r9121, r9118, r9068;
}
{
sub.f16x2 r9124, r9095, r9096;
}
{
mul.f16x2 r9127, r9124, r9072;
}
{
add.f16x2 r9130, r9121, r9127;
}
{
sub.f16x2 r9133, r9115, r9130;
}
{
add.f16x2 r9136, r9077, r9078;
}
{
mul.f16x2 r9139, r9136, r9066;
}
{
add.f16x2 r9142, r9080, r9139;
}
{
add.f16x2 r9145, r9083, r9084;
}
{
mul.f16x2 r9148, r9145, r9070;
}
{
add.f16x2 r9151, r9142, r9148;
}
{
sub.f16x2 r9154, r9089, r9090;
}
{
mul.f16x2 r9157, r9154, r9068;
}
{
sub.f16x2 r9160, r9095, r9096;
}
{
mul.f16x2 r9163, r9160, r9072;
}
{
add.f16x2 r9166, r9157, r9163;
}
{
add.f16x2 r9169, r9151, r9166;
}
{
add.f16x2 r9172, r9077, r9078;
}
{
mul.f16x2 r9175, r9172, r9070;
}
{
add.f16x2 r9178, r9080, r9175;
}
{
add.f16x2 r9181, r9083, r9084;
}
{
mul.f16x2 r9184, r9181, r9074;
}
{
add.f16x2 r9187, r9178, r9184;
}
{
sub.f16x2 r9190, r9089, r9090;
}
{
mul.f16x2 r9193, r9190, r9072;
}
{
sub.f16x2 r9196, r9095, r9096;
}
{
mul.f16x2 r9199, r9196, r9075;
}
{
add.f16x2 r9202, r9193, r9199;
}
{
sub.f16x2 r9205, r9187, r9202;
}
{
add.f16x2 r9208, r9077, r9078;
}
{
mul.f16x2 r9211, r9208, r9070;
}
{
add.f16x2 r9214, r9080, r9211;
}
{
add.f16x2 r9217, r9083, r9084;
}
{
mul.f16x2 r9220, r9217, r9074;
}
{
add.f16x2 r9223, r9214, r9220;
}
{
sub.f16x2 r9226, r9089, r9090;
}
{
mul.f16x2 r9229, r9226, r9072;
}
{
sub.f16x2 r9232, r9095, r9096;
}
{
mul.f16x2 r9235, r9232, r9075;
}
{
add.f16x2 r9238, r9229, r9235;
}
{
add.f16x2 r9241, r9223, r9238;
}
{
add.f16x2 r9244, r9089, r9090;
}
{
mul.f16x2 r9247, r9244, r9066;
}
{
add.f16x2 r9250, r9092, r9247;
}
{
add.f16x2 r9253, r9095, r9096;
}
{
mul.f16x2 r9256, r9253, r9070;
}
{
add.f16x2 r9259, r9250, r9256;
}
{
sub.f16x2 r9262, r9077, r9078;
}
{
mul.f16x2 r9265, r9262, r9068;
}
{
sub.f16x2 r9268, r9083, r9084;
}
{
mul.f16x2 r9271, r9268, r9072;
}
{
add.f16x2 r9274, r9265, r9271;
}
{
add.f16x2 r9277, r9259, r9274;
}
{
add.f16x2 r9280, r9089, r9090;
}
{
mul.f16x2 r9283, r9280, r9066;
}
{
add.f16x2 r9286, r9092, r9283;
}
{
add.f16x2 r9289, r9095, r9096;
}
{
mul.f16x2 r9292, r9289, r9070;
}
{
add.f16x2 r9295, r9286, r9292;
}
{
sub.f16x2 r9298, r9077, r9078;
}
{
mul.f16x2 r9301, r9298, r9068;
}
{
sub.f16x2 r9304, r9083, r9084;
}
{
mul.f16x2 r9307, r9304, r9072;
}
{
add.f16x2 r9310, r9301, r9307;
}
{
sub.f16x2 r9313, r9295, r9310;
}
{
add.f16x2 r9316, r9089, r9090;
}
{
mul.f16x2 r9319, r9316, r9070;
}
{
add.f16x2 r9322, r9092, r9319;
}
{
add.f16x2 r9325, r9095, r9096;
}
{
mul.f16x2 r9328, r9325, r9074;
}
{
add.f16x2 r9331, r9322, r9328;
}
{
sub.f16x2 r9334, r9077, r9078;
}
{
mul.f16x2 r9337, r9334, r9072;
}
{
sub.f16x2 r9340, r9083, r9084;
}
{
mul.f16x2 r9343, r9340, r9075;
}
{
add.f16x2 r9346, r9337, r9343;
}
{
add.f16x2 r9349, r9331, r9346;
}
{
add.f16x2 r9352, r9089, r9090;
}
{
mul.f16x2 r9355, r9352, r9070;
}
{
add.f16x2 r9358, r9092, r9355;
}
{
add.f16x2 r9361, r9095, r9096;
}
{
mul.f16x2 r9364, r9361, r9074;
}
{
add.f16x2 r9367, r9358, r9364;
}
{
sub.f16x2 r9370, r9077, r9078;
}
{
mul.f16x2 r9373, r9370, r9072;
}
{
sub.f16x2 r9376, r9083, r9084;
}
{
mul.f16x2 r9379, r9376, r9075;
}
{
add.f16x2 r9382, r9373, r9379;
}
{
sub.f16x2 r9385, r9367, r9382;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r9388, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r9389, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r9390, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r9391, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r9392, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r9393, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r9394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r9395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r9398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r9399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r9402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r9403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9404, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r9405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9410, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r9411, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r9419, {low, high};
}
{
mul.f16x2 r9436, r8167, r9388;
}
{
mul.f16x2 r9439, r8311, r9389;
}
{
sub.f16x2 r9442, r9436, r9439;
}
{
mul.f16x2 r9445, r8167, r9389;
}
{
fma.rn.f16x2 r9448, r8311, r9388, r9445;
}
{
mul.f16x2 r9452, r8489, r9390;
}
{
mul.f16x2 r9455, r8633, r9391;
}
{
sub.f16x2 r9458, r9452, r9455;
}
{
mul.f16x2 r9461, r8489, r9391;
}
{
fma.rn.f16x2 r9464, r8633, r9390, r9461;
}
{
mul.f16x2 r9468, r8811, r9392;
}
{
mul.f16x2 r9471, r8955, r9393;
}
{
sub.f16x2 r9474, r9468, r9471;
}
{
mul.f16x2 r9477, r8811, r9393;
}
{
fma.rn.f16x2 r9480, r8955, r9392, r9477;
}
{
mul.f16x2 r9484, r9133, r9394;
}
{
mul.f16x2 r9487, r9277, r9395;
}
{
sub.f16x2 r9490, r9484, r9487;
}
{
mul.f16x2 r9493, r9133, r9395;
}
{
fma.rn.f16x2 r9496, r9277, r9394, r9493;
}
{
mul.f16x2 r9500, r8239, r9390;
}
{
mul.f16x2 r9503, r8383, r9391;
}
{
sub.f16x2 r9506, r9500, r9503;
}
{
mul.f16x2 r9509, r8239, r9391;
}
{
fma.rn.f16x2 r9512, r8383, r9390, r9509;
}
{
mul.f16x2 r9516, r8561, r9394;
}
{
mul.f16x2 r9519, r8705, r9395;
}
{
sub.f16x2 r9522, r9516, r9519;
}
{
mul.f16x2 r9525, r8561, r9395;
}
{
fma.rn.f16x2 r9528, r8705, r9394, r9525;
}
{
mul.f16x2 r9532, r8883, r9398;
}
{
mul.f16x2 r9535, r9027, r9399;
}
{
sub.f16x2 r9538, r9532, r9535;
}
{
mul.f16x2 r9541, r8883, r9399;
}
{
fma.rn.f16x2 r9544, r9027, r9398, r9541;
}
{
mul.f16x2 r9548, r9205, r9402;
}
{
mul.f16x2 r9551, r9349, r9403;
}
{
sub.f16x2 r9554, r9548, r9551;
}
{
mul.f16x2 r9557, r9205, r9403;
}
{
fma.rn.f16x2 r9560, r9349, r9402, r9557;
}
{
mul.f16x2 r9564, r8275, r9392;
}
{
mul.f16x2 r9567, r8419, r9393;
}
{
sub.f16x2 r9570, r9564, r9567;
}
{
mul.f16x2 r9573, r8275, r9393;
}
{
fma.rn.f16x2 r9576, r8419, r9392, r9573;
}
{
mul.f16x2 r9580, r8597, r9398;
}
{
mul.f16x2 r9583, r8741, r9399;
}
{
sub.f16x2 r9586, r9580, r9583;
}
{
mul.f16x2 r9589, r8597, r9399;
}
{
fma.rn.f16x2 r9592, r8741, r9398, r9589;
}
{
mul.f16x2 r9596, r8919, r9404;
}
{
mul.f16x2 r9599, r9063, r9405;
}
{
sub.f16x2 r9602, r9596, r9599;
}
{
mul.f16x2 r9605, r8919, r9405;
}
{
fma.rn.f16x2 r9608, r9063, r9404, r9605;
}
{
mul.f16x2 r9612, r9241, r9410;
}
{
mul.f16x2 r9615, r9385, r9411;
}
{
sub.f16x2 r9618, r9612, r9615;
}
{
mul.f16x2 r9621, r9241, r9411;
}
{
fma.rn.f16x2 r9624, r9385, r9410, r9621;
}
{
mul.f16x2 r9628, r8203, r9394;
}
{
mul.f16x2 r9631, r8347, r9395;
}
{
sub.f16x2 r9634, r9628, r9631;
}
{
mul.f16x2 r9637, r8203, r9395;
}
{
fma.rn.f16x2 r9640, r8347, r9394, r9637;
}
{
mul.f16x2 r9644, r8525, r9402;
}
{
mul.f16x2 r9647, r8669, r9403;
}
{
sub.f16x2 r9650, r9644, r9647;
}
{
mul.f16x2 r9653, r8525, r9403;
}
{
fma.rn.f16x2 r9656, r8669, r9402, r9653;
}
{
mul.f16x2 r9660, r8847, r9410;
}
{
mul.f16x2 r9663, r8991, r9411;
}
{
sub.f16x2 r9666, r9660, r9663;
}
{
mul.f16x2 r9669, r8847, r9411;
}
{
fma.rn.f16x2 r9672, r8991, r9410, r9669;
}
{
mul.f16x2 r9676, r9169, r9418;
}
{
mul.f16x2 r9679, r9313, r9419;
}
{
sub.f16x2 r9682, r9676, r9679;
}
{
mul.f16x2 r9685, r9169, r9419;
}
{
fma.rn.f16x2 r9688, r9313, r9418, r9685;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9692, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9693, {low, high};
}
{
neg.f16x2 r9694, r9693;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9696, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9697, {low, high};
}
{
neg.f16x2 r9698, r9697;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9700, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9701, {low, high};
}
{
add.f16x2 r9702, r8119, r9085;
}
{
add.f16x2 r9705, r7797, r9702;
}
{
add.f16x2 r9708, r8441, r8763;
}
{
add.f16x2 %0, r9705, r9708;
}
{
add.f16x2 r9714, r8131, r9097;
}
{
add.f16x2 r9717, r7809, r9714;
}
{
add.f16x2 r9720, r8453, r8775;
}
{
add.f16x2 %1, r9717, r9720;
}
{
add.f16x2 r9726, r8119, r9085;
}
{
mul.f16x2 r9729, r9726, r9692;
}
{
add.f16x2 r9732, r7797, r9729;
}
{
add.f16x2 r9735, r8441, r8763;
}
{
mul.f16x2 r9738, r9735, r9696;
}
{
add.f16x2 r9741, r9732, r9738;
}
{
sub.f16x2 r9744, r8131, r9097;
}
{
mul.f16x2 r9747, r9744, r9694;
}
{
sub.f16x2 r9750, r8453, r8775;
}
{
mul.f16x2 r9753, r9750, r9698;
}
{
add.f16x2 r9756, r9747, r9753;
}
{
sub.f16x2 %10, r9741, r9756;
}
{
add.f16x2 r9762, r8119, r9085;
}
{
mul.f16x2 r9765, r9762, r9692;
}
{
add.f16x2 r9768, r7797, r9765;
}
{
add.f16x2 r9771, r8441, r8763;
}
{
mul.f16x2 r9774, r9771, r9696;
}
{
add.f16x2 r9777, r9768, r9774;
}
{
sub.f16x2 r9780, r8131, r9097;
}
{
mul.f16x2 r9783, r9780, r9694;
}
{
sub.f16x2 r9786, r8453, r8775;
}
{
mul.f16x2 r9789, r9786, r9698;
}
{
add.f16x2 r9792, r9783, r9789;
}
{
add.f16x2 %40, r9777, r9792;
}
{
add.f16x2 r9798, r8119, r9085;
}
{
mul.f16x2 r9801, r9798, r9696;
}
{
add.f16x2 r9804, r7797, r9801;
}
{
add.f16x2 r9807, r8441, r8763;
}
{
mul.f16x2 r9810, r9807, r9700;
}
{
add.f16x2 r9813, r9804, r9810;
}
{
sub.f16x2 r9816, r8131, r9097;
}
{
mul.f16x2 r9819, r9816, r9698;
}
{
sub.f16x2 r9822, r8453, r8775;
}
{
mul.f16x2 r9825, r9822, r9701;
}
{
add.f16x2 r9828, r9819, r9825;
}
{
sub.f16x2 %20, r9813, r9828;
}
{
add.f16x2 r9834, r8119, r9085;
}
{
mul.f16x2 r9837, r9834, r9696;
}
{
add.f16x2 r9840, r7797, r9837;
}
{
add.f16x2 r9843, r8441, r8763;
}
{
mul.f16x2 r9846, r9843, r9700;
}
{
add.f16x2 r9849, r9840, r9846;
}
{
sub.f16x2 r9852, r8131, r9097;
}
{
mul.f16x2 r9855, r9852, r9698;
}
{
sub.f16x2 r9858, r8453, r8775;
}
{
mul.f16x2 r9861, r9858, r9701;
}
{
add.f16x2 r9864, r9855, r9861;
}
{
add.f16x2 %30, r9849, r9864;
}
{
add.f16x2 r9870, r8131, r9097;
}
{
mul.f16x2 r9873, r9870, r9692;
}
{
add.f16x2 r9876, r7809, r9873;
}
{
add.f16x2 r9879, r8453, r8775;
}
{
mul.f16x2 r9882, r9879, r9696;
}
{
add.f16x2 r9885, r9876, r9882;
}
{
sub.f16x2 r9888, r8119, r9085;
}
{
mul.f16x2 r9891, r9888, r9694;
}
{
sub.f16x2 r9894, r8441, r8763;
}
{
mul.f16x2 r9897, r9894, r9698;
}
{
add.f16x2 r9900, r9891, r9897;
}
{
add.f16x2 %11, r9885, r9900;
}
{
add.f16x2 r9906, r8131, r9097;
}
{
mul.f16x2 r9909, r9906, r9692;
}
{
add.f16x2 r9912, r7809, r9909;
}
{
add.f16x2 r9915, r8453, r8775;
}
{
mul.f16x2 r9918, r9915, r9696;
}
{
add.f16x2 r9921, r9912, r9918;
}
{
sub.f16x2 r9924, r8119, r9085;
}
{
mul.f16x2 r9927, r9924, r9694;
}
{
sub.f16x2 r9930, r8441, r8763;
}
{
mul.f16x2 r9933, r9930, r9698;
}
{
add.f16x2 r9936, r9927, r9933;
}
{
sub.f16x2 %41, r9921, r9936;
}
{
add.f16x2 r9942, r8131, r9097;
}
{
mul.f16x2 r9945, r9942, r9696;
}
{
add.f16x2 r9948, r7809, r9945;
}
{
add.f16x2 r9951, r8453, r8775;
}
{
mul.f16x2 r9954, r9951, r9700;
}
{
add.f16x2 r9957, r9948, r9954;
}
{
sub.f16x2 r9960, r8119, r9085;
}
{
mul.f16x2 r9963, r9960, r9698;
}
{
sub.f16x2 r9966, r8441, r8763;
}
{
mul.f16x2 r9969, r9966, r9701;
}
{
add.f16x2 r9972, r9963, r9969;
}
{
add.f16x2 %21, r9957, r9972;
}
{
add.f16x2 r9978, r8131, r9097;
}
{
mul.f16x2 r9981, r9978, r9696;
}
{
add.f16x2 r9984, r7809, r9981;
}
{
add.f16x2 r9987, r8453, r8775;
}
{
mul.f16x2 r9990, r9987, r9700;
}
{
add.f16x2 r9993, r9984, r9990;
}
{
sub.f16x2 r9996, r8119, r9085;
}
{
mul.f16x2 r9999, r9996, r9698;
}
{
sub.f16x2 r10002, r8441, r8763;
}
{
mul.f16x2 r10005, r10002, r9701;
}
{
add.f16x2 r10008, r9999, r10005;
}
{
sub.f16x2 %31, r9993, r10008;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10014, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10015, {low, high};
}
{
neg.f16x2 r10016, r10015;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10019, {low, high};
}
{
neg.f16x2 r10020, r10019;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10022, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10023, {low, high};
}
{
add.f16x2 r10024, r9442, r9490;
}
{
add.f16x2 r10027, r7845, r10024;
}
{
add.f16x2 r10030, r9458, r9474;
}
{
add.f16x2 %2, r10027, r10030;
}
{
add.f16x2 r10036, r9448, r9496;
}
{
add.f16x2 r10039, r7989, r10036;
}
{
add.f16x2 r10042, r9464, r9480;
}
{
add.f16x2 %3, r10039, r10042;
}
{
add.f16x2 r10048, r9442, r9490;
}
{
mul.f16x2 r10051, r10048, r10014;
}
{
add.f16x2 r10054, r7845, r10051;
}
{
add.f16x2 r10057, r9458, r9474;
}
{
mul.f16x2 r10060, r10057, r10018;
}
{
add.f16x2 r10063, r10054, r10060;
}
{
sub.f16x2 r10066, r9448, r9496;
}
{
mul.f16x2 r10069, r10066, r10016;
}
{
sub.f16x2 r10072, r9464, r9480;
}
{
mul.f16x2 r10075, r10072, r10020;
}
{
add.f16x2 r10078, r10069, r10075;
}
{
sub.f16x2 %12, r10063, r10078;
}
{
add.f16x2 r10084, r9442, r9490;
}
{
mul.f16x2 r10087, r10084, r10014;
}
{
add.f16x2 r10090, r7845, r10087;
}
{
add.f16x2 r10093, r9458, r9474;
}
{
mul.f16x2 r10096, r10093, r10018;
}
{
add.f16x2 r10099, r10090, r10096;
}
{
sub.f16x2 r10102, r9448, r9496;
}
{
mul.f16x2 r10105, r10102, r10016;
}
{
sub.f16x2 r10108, r9464, r9480;
}
{
mul.f16x2 r10111, r10108, r10020;
}
{
add.f16x2 r10114, r10105, r10111;
}
{
add.f16x2 %42, r10099, r10114;
}
{
add.f16x2 r10120, r9442, r9490;
}
{
mul.f16x2 r10123, r10120, r10018;
}
{
add.f16x2 r10126, r7845, r10123;
}
{
add.f16x2 r10129, r9458, r9474;
}
{
mul.f16x2 r10132, r10129, r10022;
}
{
add.f16x2 r10135, r10126, r10132;
}
{
sub.f16x2 r10138, r9448, r9496;
}
{
mul.f16x2 r10141, r10138, r10020;
}
{
sub.f16x2 r10144, r9464, r9480;
}
{
mul.f16x2 r10147, r10144, r10023;
}
{
add.f16x2 r10150, r10141, r10147;
}
{
sub.f16x2 %22, r10135, r10150;
}
{
add.f16x2 r10156, r9442, r9490;
}
{
mul.f16x2 r10159, r10156, r10018;
}
{
add.f16x2 r10162, r7845, r10159;
}
{
add.f16x2 r10165, r9458, r9474;
}
{
mul.f16x2 r10168, r10165, r10022;
}
{
add.f16x2 r10171, r10162, r10168;
}
{
sub.f16x2 r10174, r9448, r9496;
}
{
mul.f16x2 r10177, r10174, r10020;
}
{
sub.f16x2 r10180, r9464, r9480;
}
{
mul.f16x2 r10183, r10180, r10023;
}
{
add.f16x2 r10186, r10177, r10183;
}
{
add.f16x2 %32, r10171, r10186;
}
{
add.f16x2 r10192, r9448, r9496;
}
{
mul.f16x2 r10195, r10192, r10014;
}
{
add.f16x2 r10198, r7989, r10195;
}
{
add.f16x2 r10201, r9464, r9480;
}
{
mul.f16x2 r10204, r10201, r10018;
}
{
add.f16x2 r10207, r10198, r10204;
}
{
sub.f16x2 r10210, r9442, r9490;
}
{
mul.f16x2 r10213, r10210, r10016;
}
{
sub.f16x2 r10216, r9458, r9474;
}
{
mul.f16x2 r10219, r10216, r10020;
}
{
add.f16x2 r10222, r10213, r10219;
}
{
add.f16x2 %13, r10207, r10222;
}
{
add.f16x2 r10228, r9448, r9496;
}
{
mul.f16x2 r10231, r10228, r10014;
}
{
add.f16x2 r10234, r7989, r10231;
}
{
add.f16x2 r10237, r9464, r9480;
}
{
mul.f16x2 r10240, r10237, r10018;
}
{
add.f16x2 r10243, r10234, r10240;
}
{
sub.f16x2 r10246, r9442, r9490;
}
{
mul.f16x2 r10249, r10246, r10016;
}
{
sub.f16x2 r10252, r9458, r9474;
}
{
mul.f16x2 r10255, r10252, r10020;
}
{
add.f16x2 r10258, r10249, r10255;
}
{
sub.f16x2 %43, r10243, r10258;
}
{
add.f16x2 r10264, r9448, r9496;
}
{
mul.f16x2 r10267, r10264, r10018;
}
{
add.f16x2 r10270, r7989, r10267;
}
{
add.f16x2 r10273, r9464, r9480;
}
{
mul.f16x2 r10276, r10273, r10022;
}
{
add.f16x2 r10279, r10270, r10276;
}
{
sub.f16x2 r10282, r9442, r9490;
}
{
mul.f16x2 r10285, r10282, r10020;
}
{
sub.f16x2 r10288, r9458, r9474;
}
{
mul.f16x2 r10291, r10288, r10023;
}
{
add.f16x2 r10294, r10285, r10291;
}
{
add.f16x2 %23, r10279, r10294;
}
{
add.f16x2 r10300, r9448, r9496;
}
{
mul.f16x2 r10303, r10300, r10018;
}
{
add.f16x2 r10306, r7989, r10303;
}
{
add.f16x2 r10309, r9464, r9480;
}
{
mul.f16x2 r10312, r10309, r10022;
}
{
add.f16x2 r10315, r10306, r10312;
}
{
sub.f16x2 r10318, r9442, r9490;
}
{
mul.f16x2 r10321, r10318, r10020;
}
{
sub.f16x2 r10324, r9458, r9474;
}
{
mul.f16x2 r10327, r10324, r10023;
}
{
add.f16x2 r10330, r10321, r10327;
}
{
sub.f16x2 %33, r10315, r10330;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10336, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10337, {low, high};
}
{
neg.f16x2 r10338, r10337;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10341, {low, high};
}
{
neg.f16x2 r10342, r10341;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10344, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10345, {low, high};
}
{
add.f16x2 r10346, r9506, r9554;
}
{
add.f16x2 r10349, r7917, r10346;
}
{
add.f16x2 r10352, r9522, r9538;
}
{
add.f16x2 %4, r10349, r10352;
}
{
add.f16x2 r10358, r9512, r9560;
}
{
add.f16x2 r10361, r8061, r10358;
}
{
add.f16x2 r10364, r9528, r9544;
}
{
add.f16x2 %5, r10361, r10364;
}
{
add.f16x2 r10370, r9506, r9554;
}
{
mul.f16x2 r10373, r10370, r10336;
}
{
add.f16x2 r10376, r7917, r10373;
}
{
add.f16x2 r10379, r9522, r9538;
}
{
mul.f16x2 r10382, r10379, r10340;
}
{
add.f16x2 r10385, r10376, r10382;
}
{
sub.f16x2 r10388, r9512, r9560;
}
{
mul.f16x2 r10391, r10388, r10338;
}
{
sub.f16x2 r10394, r9528, r9544;
}
{
mul.f16x2 r10397, r10394, r10342;
}
{
add.f16x2 r10400, r10391, r10397;
}
{
sub.f16x2 %14, r10385, r10400;
}
{
add.f16x2 r10406, r9506, r9554;
}
{
mul.f16x2 r10409, r10406, r10336;
}
{
add.f16x2 r10412, r7917, r10409;
}
{
add.f16x2 r10415, r9522, r9538;
}
{
mul.f16x2 r10418, r10415, r10340;
}
{
add.f16x2 r10421, r10412, r10418;
}
{
sub.f16x2 r10424, r9512, r9560;
}
{
mul.f16x2 r10427, r10424, r10338;
}
{
sub.f16x2 r10430, r9528, r9544;
}
{
mul.f16x2 r10433, r10430, r10342;
}
{
add.f16x2 r10436, r10427, r10433;
}
{
add.f16x2 %44, r10421, r10436;
}
{
add.f16x2 r10442, r9506, r9554;
}
{
mul.f16x2 r10445, r10442, r10340;
}
{
add.f16x2 r10448, r7917, r10445;
}
{
add.f16x2 r10451, r9522, r9538;
}
{
mul.f16x2 r10454, r10451, r10344;
}
{
add.f16x2 r10457, r10448, r10454;
}
{
sub.f16x2 r10460, r9512, r9560;
}
{
mul.f16x2 r10463, r10460, r10342;
}
{
sub.f16x2 r10466, r9528, r9544;
}
{
mul.f16x2 r10469, r10466, r10345;
}
{
add.f16x2 r10472, r10463, r10469;
}
{
sub.f16x2 %24, r10457, r10472;
}
{
add.f16x2 r10478, r9506, r9554;
}
{
mul.f16x2 r10481, r10478, r10340;
}
{
add.f16x2 r10484, r7917, r10481;
}
{
add.f16x2 r10487, r9522, r9538;
}
{
mul.f16x2 r10490, r10487, r10344;
}
{
add.f16x2 r10493, r10484, r10490;
}
{
sub.f16x2 r10496, r9512, r9560;
}
{
mul.f16x2 r10499, r10496, r10342;
}
{
sub.f16x2 r10502, r9528, r9544;
}
{
mul.f16x2 r10505, r10502, r10345;
}
{
add.f16x2 r10508, r10499, r10505;
}
{
add.f16x2 %34, r10493, r10508;
}
{
add.f16x2 r10514, r9512, r9560;
}
{
mul.f16x2 r10517, r10514, r10336;
}
{
add.f16x2 r10520, r8061, r10517;
}
{
add.f16x2 r10523, r9528, r9544;
}
{
mul.f16x2 r10526, r10523, r10340;
}
{
add.f16x2 r10529, r10520, r10526;
}
{
sub.f16x2 r10532, r9506, r9554;
}
{
mul.f16x2 r10535, r10532, r10338;
}
{
sub.f16x2 r10538, r9522, r9538;
}
{
mul.f16x2 r10541, r10538, r10342;
}
{
add.f16x2 r10544, r10535, r10541;
}
{
add.f16x2 %15, r10529, r10544;
}
{
add.f16x2 r10550, r9512, r9560;
}
{
mul.f16x2 r10553, r10550, r10336;
}
{
add.f16x2 r10556, r8061, r10553;
}
{
add.f16x2 r10559, r9528, r9544;
}
{
mul.f16x2 r10562, r10559, r10340;
}
{
add.f16x2 r10565, r10556, r10562;
}
{
sub.f16x2 r10568, r9506, r9554;
}
{
mul.f16x2 r10571, r10568, r10338;
}
{
sub.f16x2 r10574, r9522, r9538;
}
{
mul.f16x2 r10577, r10574, r10342;
}
{
add.f16x2 r10580, r10571, r10577;
}
{
sub.f16x2 %45, r10565, r10580;
}
{
add.f16x2 r10586, r9512, r9560;
}
{
mul.f16x2 r10589, r10586, r10340;
}
{
add.f16x2 r10592, r8061, r10589;
}
{
add.f16x2 r10595, r9528, r9544;
}
{
mul.f16x2 r10598, r10595, r10344;
}
{
add.f16x2 r10601, r10592, r10598;
}
{
sub.f16x2 r10604, r9506, r9554;
}
{
mul.f16x2 r10607, r10604, r10342;
}
{
sub.f16x2 r10610, r9522, r9538;
}
{
mul.f16x2 r10613, r10610, r10345;
}
{
add.f16x2 r10616, r10607, r10613;
}
{
add.f16x2 %25, r10601, r10616;
}
{
add.f16x2 r10622, r9512, r9560;
}
{
mul.f16x2 r10625, r10622, r10340;
}
{
add.f16x2 r10628, r8061, r10625;
}
{
add.f16x2 r10631, r9528, r9544;
}
{
mul.f16x2 r10634, r10631, r10344;
}
{
add.f16x2 r10637, r10628, r10634;
}
{
sub.f16x2 r10640, r9506, r9554;
}
{
mul.f16x2 r10643, r10640, r10342;
}
{
sub.f16x2 r10646, r9522, r9538;
}
{
mul.f16x2 r10649, r10646, r10345;
}
{
add.f16x2 r10652, r10643, r10649;
}
{
sub.f16x2 %35, r10637, r10652;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10658, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10659, {low, high};
}
{
neg.f16x2 r10660, r10659;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10662, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10663, {low, high};
}
{
neg.f16x2 r10664, r10663;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10666, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10667, {low, high};
}
{
add.f16x2 r10668, r9570, r9618;
}
{
add.f16x2 r10671, r7953, r10668;
}
{
add.f16x2 r10674, r9586, r9602;
}
{
add.f16x2 %6, r10671, r10674;
}
{
add.f16x2 r10680, r9576, r9624;
}
{
add.f16x2 r10683, r8097, r10680;
}
{
add.f16x2 r10686, r9592, r9608;
}
{
add.f16x2 %7, r10683, r10686;
}
{
add.f16x2 r10692, r9570, r9618;
}
{
mul.f16x2 r10695, r10692, r10658;
}
{
add.f16x2 r10698, r7953, r10695;
}
{
add.f16x2 r10701, r9586, r9602;
}
{
mul.f16x2 r10704, r10701, r10662;
}
{
add.f16x2 r10707, r10698, r10704;
}
{
sub.f16x2 r10710, r9576, r9624;
}
{
mul.f16x2 r10713, r10710, r10660;
}
{
sub.f16x2 r10716, r9592, r9608;
}
{
mul.f16x2 r10719, r10716, r10664;
}
{
add.f16x2 r10722, r10713, r10719;
}
{
sub.f16x2 %16, r10707, r10722;
}
{
add.f16x2 r10728, r9570, r9618;
}
{
mul.f16x2 r10731, r10728, r10658;
}
{
add.f16x2 r10734, r7953, r10731;
}
{
add.f16x2 r10737, r9586, r9602;
}
{
mul.f16x2 r10740, r10737, r10662;
}
{
add.f16x2 r10743, r10734, r10740;
}
{
sub.f16x2 r10746, r9576, r9624;
}
{
mul.f16x2 r10749, r10746, r10660;
}
{
sub.f16x2 r10752, r9592, r9608;
}
{
mul.f16x2 r10755, r10752, r10664;
}
{
add.f16x2 r10758, r10749, r10755;
}
{
add.f16x2 %46, r10743, r10758;
}
{
add.f16x2 r10764, r9570, r9618;
}
{
mul.f16x2 r10767, r10764, r10662;
}
{
add.f16x2 r10770, r7953, r10767;
}
{
add.f16x2 r10773, r9586, r9602;
}
{
mul.f16x2 r10776, r10773, r10666;
}
{
add.f16x2 r10779, r10770, r10776;
}
{
sub.f16x2 r10782, r9576, r9624;
}
{
mul.f16x2 r10785, r10782, r10664;
}
{
sub.f16x2 r10788, r9592, r9608;
}
{
mul.f16x2 r10791, r10788, r10667;
}
{
add.f16x2 r10794, r10785, r10791;
}
{
sub.f16x2 %26, r10779, r10794;
}
{
add.f16x2 r10800, r9570, r9618;
}
{
mul.f16x2 r10803, r10800, r10662;
}
{
add.f16x2 r10806, r7953, r10803;
}
{
add.f16x2 r10809, r9586, r9602;
}
{
mul.f16x2 r10812, r10809, r10666;
}
{
add.f16x2 r10815, r10806, r10812;
}
{
sub.f16x2 r10818, r9576, r9624;
}
{
mul.f16x2 r10821, r10818, r10664;
}
{
sub.f16x2 r10824, r9592, r9608;
}
{
mul.f16x2 r10827, r10824, r10667;
}
{
add.f16x2 r10830, r10821, r10827;
}
{
add.f16x2 %36, r10815, r10830;
}
{
add.f16x2 r10836, r9576, r9624;
}
{
mul.f16x2 r10839, r10836, r10658;
}
{
add.f16x2 r10842, r8097, r10839;
}
{
add.f16x2 r10845, r9592, r9608;
}
{
mul.f16x2 r10848, r10845, r10662;
}
{
add.f16x2 r10851, r10842, r10848;
}
{
sub.f16x2 r10854, r9570, r9618;
}
{
mul.f16x2 r10857, r10854, r10660;
}
{
sub.f16x2 r10860, r9586, r9602;
}
{
mul.f16x2 r10863, r10860, r10664;
}
{
add.f16x2 r10866, r10857, r10863;
}
{
add.f16x2 %17, r10851, r10866;
}
{
add.f16x2 r10872, r9576, r9624;
}
{
mul.f16x2 r10875, r10872, r10658;
}
{
add.f16x2 r10878, r8097, r10875;
}
{
add.f16x2 r10881, r9592, r9608;
}
{
mul.f16x2 r10884, r10881, r10662;
}
{
add.f16x2 r10887, r10878, r10884;
}
{
sub.f16x2 r10890, r9570, r9618;
}
{
mul.f16x2 r10893, r10890, r10660;
}
{
sub.f16x2 r10896, r9586, r9602;
}
{
mul.f16x2 r10899, r10896, r10664;
}
{
add.f16x2 r10902, r10893, r10899;
}
{
sub.f16x2 %47, r10887, r10902;
}
{
add.f16x2 r10908, r9576, r9624;
}
{
mul.f16x2 r10911, r10908, r10662;
}
{
add.f16x2 r10914, r8097, r10911;
}
{
add.f16x2 r10917, r9592, r9608;
}
{
mul.f16x2 r10920, r10917, r10666;
}
{
add.f16x2 r10923, r10914, r10920;
}
{
sub.f16x2 r10926, r9570, r9618;
}
{
mul.f16x2 r10929, r10926, r10664;
}
{
sub.f16x2 r10932, r9586, r9602;
}
{
mul.f16x2 r10935, r10932, r10667;
}
{
add.f16x2 r10938, r10929, r10935;
}
{
add.f16x2 %27, r10923, r10938;
}
{
add.f16x2 r10944, r9576, r9624;
}
{
mul.f16x2 r10947, r10944, r10662;
}
{
add.f16x2 r10950, r8097, r10947;
}
{
add.f16x2 r10953, r9592, r9608;
}
{
mul.f16x2 r10956, r10953, r10666;
}
{
add.f16x2 r10959, r10950, r10956;
}
{
sub.f16x2 r10962, r9570, r9618;
}
{
mul.f16x2 r10965, r10962, r10664;
}
{
sub.f16x2 r10968, r9586, r9602;
}
{
mul.f16x2 r10971, r10968, r10667;
}
{
add.f16x2 r10974, r10965, r10971;
}
{
sub.f16x2 %37, r10959, r10974;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10980, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10981, {low, high};
}
{
neg.f16x2 r10982, r10981;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10984, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10985, {low, high};
}
{
neg.f16x2 r10986, r10985;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10988, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10989, {low, high};
}
{
add.f16x2 r10990, r9634, r9682;
}
{
add.f16x2 r10993, r7881, r10990;
}
{
add.f16x2 r10996, r9650, r9666;
}
{
add.f16x2 %8, r10993, r10996;
}
{
add.f16x2 r11002, r9640, r9688;
}
{
add.f16x2 r11005, r8025, r11002;
}
{
add.f16x2 r11008, r9656, r9672;
}
{
add.f16x2 %9, r11005, r11008;
}
{
add.f16x2 r11014, r9634, r9682;
}
{
mul.f16x2 r11017, r11014, r10980;
}
{
add.f16x2 r11020, r7881, r11017;
}
{
add.f16x2 r11023, r9650, r9666;
}
{
mul.f16x2 r11026, r11023, r10984;
}
{
add.f16x2 r11029, r11020, r11026;
}
{
sub.f16x2 r11032, r9640, r9688;
}
{
mul.f16x2 r11035, r11032, r10982;
}
{
sub.f16x2 r11038, r9656, r9672;
}
{
mul.f16x2 r11041, r11038, r10986;
}
{
add.f16x2 r11044, r11035, r11041;
}
{
sub.f16x2 %18, r11029, r11044;
}
{
add.f16x2 r11050, r9634, r9682;
}
{
mul.f16x2 r11053, r11050, r10980;
}
{
add.f16x2 r11056, r7881, r11053;
}
{
add.f16x2 r11059, r9650, r9666;
}
{
mul.f16x2 r11062, r11059, r10984;
}
{
add.f16x2 r11065, r11056, r11062;
}
{
sub.f16x2 r11068, r9640, r9688;
}
{
mul.f16x2 r11071, r11068, r10982;
}
{
sub.f16x2 r11074, r9656, r9672;
}
{
mul.f16x2 r11077, r11074, r10986;
}
{
add.f16x2 r11080, r11071, r11077;
}
{
add.f16x2 %48, r11065, r11080;
}
{
add.f16x2 r11086, r9634, r9682;
}
{
mul.f16x2 r11089, r11086, r10984;
}
{
add.f16x2 r11092, r7881, r11089;
}
{
add.f16x2 r11095, r9650, r9666;
}
{
mul.f16x2 r11098, r11095, r10988;
}
{
add.f16x2 r11101, r11092, r11098;
}
{
sub.f16x2 r11104, r9640, r9688;
}
{
mul.f16x2 r11107, r11104, r10986;
}
{
sub.f16x2 r11110, r9656, r9672;
}
{
mul.f16x2 r11113, r11110, r10989;
}
{
add.f16x2 r11116, r11107, r11113;
}
{
sub.f16x2 %28, r11101, r11116;
}
{
add.f16x2 r11122, r9634, r9682;
}
{
mul.f16x2 r11125, r11122, r10984;
}
{
add.f16x2 r11128, r7881, r11125;
}
{
add.f16x2 r11131, r9650, r9666;
}
{
mul.f16x2 r11134, r11131, r10988;
}
{
add.f16x2 r11137, r11128, r11134;
}
{
sub.f16x2 r11140, r9640, r9688;
}
{
mul.f16x2 r11143, r11140, r10986;
}
{
sub.f16x2 r11146, r9656, r9672;
}
{
mul.f16x2 r11149, r11146, r10989;
}
{
add.f16x2 r11152, r11143, r11149;
}
{
add.f16x2 %38, r11137, r11152;
}
{
add.f16x2 r11158, r9640, r9688;
}
{
mul.f16x2 r11161, r11158, r10980;
}
{
add.f16x2 r11164, r8025, r11161;
}
{
add.f16x2 r11167, r9656, r9672;
}
{
mul.f16x2 r11170, r11167, r10984;
}
{
add.f16x2 r11173, r11164, r11170;
}
{
sub.f16x2 r11176, r9634, r9682;
}
{
mul.f16x2 r11179, r11176, r10982;
}
{
sub.f16x2 r11182, r9650, r9666;
}
{
mul.f16x2 r11185, r11182, r10986;
}
{
add.f16x2 r11188, r11179, r11185;
}
{
add.f16x2 %19, r11173, r11188;
}
{
add.f16x2 r11194, r9640, r9688;
}
{
mul.f16x2 r11197, r11194, r10980;
}
{
add.f16x2 r11200, r8025, r11197;
}
{
add.f16x2 r11203, r9656, r9672;
}
{
mul.f16x2 r11206, r11203, r10984;
}
{
add.f16x2 r11209, r11200, r11206;
}
{
sub.f16x2 r11212, r9634, r9682;
}
{
mul.f16x2 r11215, r11212, r10982;
}
{
sub.f16x2 r11218, r9650, r9666;
}
{
mul.f16x2 r11221, r11218, r10986;
}
{
add.f16x2 r11224, r11215, r11221;
}
{
sub.f16x2 %49, r11209, r11224;
}
{
add.f16x2 r11230, r9640, r9688;
}
{
mul.f16x2 r11233, r11230, r10984;
}
{
add.f16x2 r11236, r8025, r11233;
}
{
add.f16x2 r11239, r9656, r9672;
}
{
mul.f16x2 r11242, r11239, r10988;
}
{
add.f16x2 r11245, r11236, r11242;
}
{
sub.f16x2 r11248, r9634, r9682;
}
{
mul.f16x2 r11251, r11248, r10986;
}
{
sub.f16x2 r11254, r9650, r9666;
}
{
mul.f16x2 r11257, r11254, r10989;
}
{
add.f16x2 r11260, r11251, r11257;
}
{
add.f16x2 %29, r11245, r11260;
}
{
add.f16x2 r11266, r9640, r9688;
}
{
mul.f16x2 r11269, r11266, r10984;
}
{
add.f16x2 r11272, r8025, r11269;
}
{
add.f16x2 r11275, r9656, r9672;
}
{
mul.f16x2 r11278, r11275, r10988;
}
{
add.f16x2 r11281, r11272, r11278;
}
{
sub.f16x2 r11284, r9634, r9682;
}
{
mul.f16x2 r11287, r11284, r10986;
}
{
sub.f16x2 r11290, r9650, r9666;
}
{
mul.f16x2 r11293, r11290, r10989;
}
{
add.f16x2 r11296, r11287, r11293;
}
{
sub.f16x2 %39, r11281, r11296;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1118, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.local .align 8 .b8 __local_depot1[200];
.reg .b64 SP;
.reg .b64 SPL;
.reg .pred p<3>;
.reg .f32 f<695>;
.reg .b32 r<11314>;
.reg .b64 rd<19>;
mov.u64 SPL, __local_depot1;
add.u64 rd3, SPL, 0;
mov.u32 r3551, %tid.y;
mul.lo.s32 r1, r3551, 15625;
add.s64 rd4, rd3, 4;
mov.f32 f214, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r23, {low, high};
}
mov.f32 f216, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r24, {low, high};
}
{
neg.f16x2 r25, r24;
}
mov.f32 f210, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r27, {low, high};
}
mov.f32 f212, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r28, {low, high};
}
{
neg.f16x2 r29, r28;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r31, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r32, {low, high};
}
{
add.f16x2 r33, %61, %91;
}
{
add.f16x2 r36, %51, r33;
}
{
add.f16x2 r39, %71, %81;
}
{
add.f16x2 r42, r36, r39;
}
{
add.f16x2 r45, %62, %92;
}
{
add.f16x2 r48, %52, r45;
}
{
add.f16x2 r51, %72, %82;
}
{
add.f16x2 r54, r48, r51;
}
{
add.f16x2 r57, %61, %91;
}
{
mul.f16x2 r60, r57, r23;
}
{
add.f16x2 r63, %51, r60;
}
{
add.f16x2 r66, %71, %81;
}
{
mul.f16x2 r69, r66, r27;
}
{
add.f16x2 r72, r63, r69;
}
{
sub.f16x2 r75, %62, %92;
}
{
mul.f16x2 r78, r75, r25;
}
{
sub.f16x2 r81, %72, %82;
}
{
mul.f16x2 r84, r81, r29;
}
{
add.f16x2 r87, r78, r84;
}
{
sub.f16x2 r90, r72, r87;
}
{
add.f16x2 r93, %61, %91;
}
{
mul.f16x2 r96, r93, r23;
}
{
add.f16x2 r99, %51, r96;
}
{
add.f16x2 r102, %71, %81;
}
{
mul.f16x2 r105, r102, r27;
}
{
add.f16x2 r108, r99, r105;
}
{
sub.f16x2 r111, %62, %92;
}
{
mul.f16x2 r114, r111, r25;
}
{
sub.f16x2 r117, %72, %82;
}
{
mul.f16x2 r120, r117, r29;
}
{
add.f16x2 r123, r114, r120;
}
{
add.f16x2 r126, r108, r123;
}
{
add.f16x2 r129, %61, %91;
}
{
mul.f16x2 r132, r129, r27;
}
{
add.f16x2 r135, %51, r132;
}
{
add.f16x2 r138, %71, %81;
}
{
mul.f16x2 r141, r138, r31;
}
{
add.f16x2 r144, r135, r141;
}
{
sub.f16x2 r147, %62, %92;
}
{
mul.f16x2 r150, r147, r29;
}
{
sub.f16x2 r153, %72, %82;
}
{
mul.f16x2 r156, r153, r32;
}
{
add.f16x2 r159, r150, r156;
}
{
sub.f16x2 r162, r144, r159;
}
{
add.f16x2 r165, %61, %91;
}
{
mul.f16x2 r168, r165, r27;
}
{
add.f16x2 r171, %51, r168;
}
{
add.f16x2 r174, %71, %81;
}
{
mul.f16x2 r177, r174, r31;
}
{
add.f16x2 r180, r171, r177;
}
{
sub.f16x2 r183, %62, %92;
}
{
mul.f16x2 r186, r183, r29;
}
{
sub.f16x2 r189, %72, %82;
}
{
mul.f16x2 r192, r189, r32;
}
{
add.f16x2 r195, r186, r192;
}
{
add.f16x2 r198, r180, r195;
}
{
add.f16x2 r201, %62, %92;
}
{
mul.f16x2 r204, r201, r23;
}
{
add.f16x2 r207, %52, r204;
}
{
add.f16x2 r210, %72, %82;
}
{
mul.f16x2 r213, r210, r27;
}
{
add.f16x2 r216, r207, r213;
}
{
sub.f16x2 r219, %61, %91;
}
{
mul.f16x2 r222, r219, r25;
}
{
sub.f16x2 r225, %71, %81;
}
{
mul.f16x2 r228, r225, r29;
}
{
add.f16x2 r231, r222, r228;
}
{
add.f16x2 r234, r216, r231;
}
{
add.f16x2 r237, %62, %92;
}
{
mul.f16x2 r240, r237, r23;
}
{
add.f16x2 r243, %52, r240;
}
{
add.f16x2 r246, %72, %82;
}
{
mul.f16x2 r249, r246, r27;
}
{
add.f16x2 r252, r243, r249;
}
{
sub.f16x2 r255, %61, %91;
}
{
mul.f16x2 r258, r255, r25;
}
{
sub.f16x2 r261, %71, %81;
}
{
mul.f16x2 r264, r261, r29;
}
{
add.f16x2 r267, r258, r264;
}
{
sub.f16x2 r270, r252, r267;
}
{
add.f16x2 r273, %62, %92;
}
{
mul.f16x2 r276, r273, r27;
}
{
add.f16x2 r279, %52, r276;
}
{
add.f16x2 r282, %72, %82;
}
{
mul.f16x2 r285, r282, r31;
}
{
add.f16x2 r288, r279, r285;
}
{
sub.f16x2 r291, %61, %91;
}
{
mul.f16x2 r294, r291, r29;
}
{
sub.f16x2 r297, %71, %81;
}
{
mul.f16x2 r300, r297, r32;
}
{
add.f16x2 r303, r294, r300;
}
{
add.f16x2 r306, r288, r303;
}
{
add.f16x2 r309, %62, %92;
}
{
mul.f16x2 r312, r309, r27;
}
{
add.f16x2 r315, %52, r312;
}
{
add.f16x2 r318, %72, %82;
}
{
mul.f16x2 r321, r318, r31;
}
{
add.f16x2 r324, r315, r321;
}
{
sub.f16x2 r327, %61, %91;
}
{
mul.f16x2 r330, r327, r29;
}
{
sub.f16x2 r333, %71, %81;
}
{
mul.f16x2 r336, r333, r32;
}
{
add.f16x2 r339, r330, r336;
}
{
sub.f16x2 r342, r324, r339;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r346, {low, high};
}
{
neg.f16x2 r347, r346;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r349, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r350, {low, high};
}
{
neg.f16x2 r351, r350;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r353, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r354, {low, high};
}
{
add.f16x2 r355, %63, %93;
}
{
add.f16x2 r358, %53, r355;
}
{
add.f16x2 r361, %73, %83;
}
{
add.f16x2 r364, r358, r361;
}
{
add.f16x2 r367, %64, %94;
}
{
add.f16x2 r370, %54, r367;
}
{
add.f16x2 r373, %74, %84;
}
{
add.f16x2 r376, r370, r373;
}
{
add.f16x2 r379, %63, %93;
}
{
mul.f16x2 r382, r379, r345;
}
{
add.f16x2 r385, %53, r382;
}
{
add.f16x2 r388, %73, %83;
}
{
mul.f16x2 r391, r388, r349;
}
{
add.f16x2 r394, r385, r391;
}
{
sub.f16x2 r397, %64, %94;
}
{
mul.f16x2 r400, r397, r347;
}
{
sub.f16x2 r403, %74, %84;
}
{
mul.f16x2 r406, r403, r351;
}
{
add.f16x2 r409, r400, r406;
}
{
sub.f16x2 r412, r394, r409;
}
{
add.f16x2 r415, %63, %93;
}
{
mul.f16x2 r418, r415, r345;
}
{
add.f16x2 r421, %53, r418;
}
{
add.f16x2 r424, %73, %83;
}
{
mul.f16x2 r427, r424, r349;
}
{
add.f16x2 r430, r421, r427;
}
{
sub.f16x2 r433, %64, %94;
}
{
mul.f16x2 r436, r433, r347;
}
{
sub.f16x2 r439, %74, %84;
}
{
mul.f16x2 r442, r439, r351;
}
{
add.f16x2 r445, r436, r442;
}
{
add.f16x2 r448, r430, r445;
}
{
add.f16x2 r451, %63, %93;
}
{
mul.f16x2 r454, r451, r349;
}
{
add.f16x2 r457, %53, r454;
}
{
add.f16x2 r460, %73, %83;
}
{
mul.f16x2 r463, r460, r353;
}
{
add.f16x2 r466, r457, r463;
}
{
sub.f16x2 r469, %64, %94;
}
{
mul.f16x2 r472, r469, r351;
}
{
sub.f16x2 r475, %74, %84;
}
{
mul.f16x2 r478, r475, r354;
}
{
add.f16x2 r481, r472, r478;
}
{
sub.f16x2 r484, r466, r481;
}
{
add.f16x2 r487, %63, %93;
}
{
mul.f16x2 r490, r487, r349;
}
{
add.f16x2 r493, %53, r490;
}
{
add.f16x2 r496, %73, %83;
}
{
mul.f16x2 r499, r496, r353;
}
{
add.f16x2 r502, r493, r499;
}
{
sub.f16x2 r505, %64, %94;
}
{
mul.f16x2 r508, r505, r351;
}
{
sub.f16x2 r511, %74, %84;
}
{
mul.f16x2 r514, r511, r354;
}
{
add.f16x2 r517, r508, r514;
}
{
add.f16x2 r520, r502, r517;
}
{
add.f16x2 r523, %64, %94;
}
{
mul.f16x2 r526, r523, r345;
}
{
add.f16x2 r529, %54, r526;
}
{
add.f16x2 r532, %74, %84;
}
{
mul.f16x2 r535, r532, r349;
}
{
add.f16x2 r538, r529, r535;
}
{
sub.f16x2 r541, %63, %93;
}
{
mul.f16x2 r544, r541, r347;
}
{
sub.f16x2 r547, %73, %83;
}
{
mul.f16x2 r550, r547, r351;
}
{
add.f16x2 r553, r544, r550;
}
{
add.f16x2 r556, r538, r553;
}
{
add.f16x2 r559, %64, %94;
}
{
mul.f16x2 r562, r559, r345;
}
{
add.f16x2 r565, %54, r562;
}
{
add.f16x2 r568, %74, %84;
}
{
mul.f16x2 r571, r568, r349;
}
{
add.f16x2 r574, r565, r571;
}
{
sub.f16x2 r577, %63, %93;
}
{
mul.f16x2 r580, r577, r347;
}
{
sub.f16x2 r583, %73, %83;
}
{
mul.f16x2 r586, r583, r351;
}
{
add.f16x2 r589, r580, r586;
}
{
sub.f16x2 r592, r574, r589;
}
{
add.f16x2 r595, %64, %94;
}
{
mul.f16x2 r598, r595, r349;
}
{
add.f16x2 r601, %54, r598;
}
{
add.f16x2 r604, %74, %84;
}
{
mul.f16x2 r607, r604, r353;
}
{
add.f16x2 r610, r601, r607;
}
{
sub.f16x2 r613, %63, %93;
}
{
mul.f16x2 r616, r613, r351;
}
{
sub.f16x2 r619, %73, %83;
}
{
mul.f16x2 r622, r619, r354;
}
{
add.f16x2 r625, r616, r622;
}
{
add.f16x2 r628, r610, r625;
}
{
add.f16x2 r631, %64, %94;
}
{
mul.f16x2 r634, r631, r349;
}
{
add.f16x2 r637, %54, r634;
}
{
add.f16x2 r640, %74, %84;
}
{
mul.f16x2 r643, r640, r353;
}
{
add.f16x2 r646, r637, r643;
}
{
sub.f16x2 r649, %63, %93;
}
{
mul.f16x2 r652, r649, r351;
}
{
sub.f16x2 r655, %73, %83;
}
{
mul.f16x2 r658, r655, r354;
}
{
add.f16x2 r661, r652, r658;
}
{
sub.f16x2 r664, r646, r661;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r667, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r668, {low, high};
}
{
neg.f16x2 r669, r668;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r671, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r672, {low, high};
}
{
neg.f16x2 r673, r672;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r675, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r676, {low, high};
}
{
add.f16x2 r677, %65, %95;
}
{
add.f16x2 r680, %55, r677;
}
{
add.f16x2 r683, %75, %85;
}
{
add.f16x2 r686, r680, r683;
}
{
add.f16x2 r689, %66, %96;
}
{
add.f16x2 r692, %56, r689;
}
{
add.f16x2 r695, %76, %86;
}
{
add.f16x2 r698, r692, r695;
}
{
add.f16x2 r701, %65, %95;
}
{
mul.f16x2 r704, r701, r667;
}
{
add.f16x2 r707, %55, r704;
}
{
add.f16x2 r710, %75, %85;
}
{
mul.f16x2 r713, r710, r671;
}
{
add.f16x2 r716, r707, r713;
}
{
sub.f16x2 r719, %66, %96;
}
{
mul.f16x2 r722, r719, r669;
}
{
sub.f16x2 r725, %76, %86;
}
{
mul.f16x2 r728, r725, r673;
}
{
add.f16x2 r731, r722, r728;
}
{
sub.f16x2 r734, r716, r731;
}
{
add.f16x2 r737, %65, %95;
}
{
mul.f16x2 r740, r737, r667;
}
{
add.f16x2 r743, %55, r740;
}
{
add.f16x2 r746, %75, %85;
}
{
mul.f16x2 r749, r746, r671;
}
{
add.f16x2 r752, r743, r749;
}
{
sub.f16x2 r755, %66, %96;
}
{
mul.f16x2 r758, r755, r669;
}
{
sub.f16x2 r761, %76, %86;
}
{
mul.f16x2 r764, r761, r673;
}
{
add.f16x2 r767, r758, r764;
}
{
add.f16x2 r770, r752, r767;
}
{
add.f16x2 r773, %65, %95;
}
{
mul.f16x2 r776, r773, r671;
}
{
add.f16x2 r779, %55, r776;
}
{
add.f16x2 r782, %75, %85;
}
{
mul.f16x2 r785, r782, r675;
}
{
add.f16x2 r788, r779, r785;
}
{
sub.f16x2 r791, %66, %96;
}
{
mul.f16x2 r794, r791, r673;
}
{
sub.f16x2 r797, %76, %86;
}
{
mul.f16x2 r800, r797, r676;
}
{
add.f16x2 r803, r794, r800;
}
{
sub.f16x2 r806, r788, r803;
}
{
add.f16x2 r809, %65, %95;
}
{
mul.f16x2 r812, r809, r671;
}
{
add.f16x2 r815, %55, r812;
}
{
add.f16x2 r818, %75, %85;
}
{
mul.f16x2 r821, r818, r675;
}
{
add.f16x2 r824, r815, r821;
}
{
sub.f16x2 r827, %66, %96;
}
{
mul.f16x2 r830, r827, r673;
}
{
sub.f16x2 r833, %76, %86;
}
{
mul.f16x2 r836, r833, r676;
}
{
add.f16x2 r839, r830, r836;
}
{
add.f16x2 r842, r824, r839;
}
{
add.f16x2 r845, %66, %96;
}
{
mul.f16x2 r848, r845, r667;
}
{
add.f16x2 r851, %56, r848;
}
{
add.f16x2 r854, %76, %86;
}
{
mul.f16x2 r857, r854, r671;
}
{
add.f16x2 r860, r851, r857;
}
{
sub.f16x2 r863, %65, %95;
}
{
mul.f16x2 r866, r863, r669;
}
{
sub.f16x2 r869, %75, %85;
}
{
mul.f16x2 r872, r869, r673;
}
{
add.f16x2 r875, r866, r872;
}
{
add.f16x2 r878, r860, r875;
}
{
add.f16x2 r881, %66, %96;
}
{
mul.f16x2 r884, r881, r667;
}
{
add.f16x2 r887, %56, r884;
}
{
add.f16x2 r890, %76, %86;
}
{
mul.f16x2 r893, r890, r671;
}
{
add.f16x2 r896, r887, r893;
}
{
sub.f16x2 r899, %65, %95;
}
{
mul.f16x2 r902, r899, r669;
}
{
sub.f16x2 r905, %75, %85;
}
{
mul.f16x2 r908, r905, r673;
}
{
add.f16x2 r911, r902, r908;
}
{
sub.f16x2 r914, r896, r911;
}
{
add.f16x2 r917, %66, %96;
}
{
mul.f16x2 r920, r917, r671;
}
{
add.f16x2 r923, %56, r920;
}
{
add.f16x2 r926, %76, %86;
}
{
mul.f16x2 r929, r926, r675;
}
{
add.f16x2 r932, r923, r929;
}
{
sub.f16x2 r935, %65, %95;
}
{
mul.f16x2 r938, r935, r673;
}
{
sub.f16x2 r941, %75, %85;
}
{
mul.f16x2 r944, r941, r676;
}
{
add.f16x2 r947, r938, r944;
}
{
add.f16x2 r950, r932, r947;
}
{
add.f16x2 r953, %66, %96;
}
{
mul.f16x2 r956, r953, r671;
}
{
add.f16x2 r959, %56, r956;
}
{
add.f16x2 r962, %76, %86;
}
{
mul.f16x2 r965, r962, r675;
}
{
add.f16x2 r968, r959, r965;
}
{
sub.f16x2 r971, %65, %95;
}
{
mul.f16x2 r974, r971, r673;
}
{
sub.f16x2 r977, %75, %85;
}
{
mul.f16x2 r980, r977, r676;
}
{
add.f16x2 r983, r974, r980;
}
{
sub.f16x2 r986, r968, r983;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r989, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r990, {low, high};
}
{
neg.f16x2 r991, r990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r993, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r994, {low, high};
}
{
neg.f16x2 r995, r994;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r998, {low, high};
}
{
add.f16x2 r999, %67, %97;
}
{
add.f16x2 r1002, %57, r999;
}
{
add.f16x2 r1005, %77, %87;
}
{
add.f16x2 r1008, r1002, r1005;
}
{
add.f16x2 r1011, %68, %98;
}
{
add.f16x2 r1014, %58, r1011;
}
{
add.f16x2 r1017, %78, %88;
}
{
add.f16x2 r1020, r1014, r1017;
}
{
add.f16x2 r1023, %67, %97;
}
{
mul.f16x2 r1026, r1023, r989;
}
{
add.f16x2 r1029, %57, r1026;
}
{
add.f16x2 r1032, %77, %87;
}
{
mul.f16x2 r1035, r1032, r993;
}
{
add.f16x2 r1038, r1029, r1035;
}
{
sub.f16x2 r1041, %68, %98;
}
{
mul.f16x2 r1044, r1041, r991;
}
{
sub.f16x2 r1047, %78, %88;
}
{
mul.f16x2 r1050, r1047, r995;
}
{
add.f16x2 r1053, r1044, r1050;
}
{
sub.f16x2 r1056, r1038, r1053;
}
{
add.f16x2 r1059, %67, %97;
}
{
mul.f16x2 r1062, r1059, r989;
}
{
add.f16x2 r1065, %57, r1062;
}
{
add.f16x2 r1068, %77, %87;
}
{
mul.f16x2 r1071, r1068, r993;
}
{
add.f16x2 r1074, r1065, r1071;
}
{
sub.f16x2 r1077, %68, %98;
}
{
mul.f16x2 r1080, r1077, r991;
}
{
sub.f16x2 r1083, %78, %88;
}
{
mul.f16x2 r1086, r1083, r995;
}
{
add.f16x2 r1089, r1080, r1086;
}
{
add.f16x2 r1092, r1074, r1089;
}
{
add.f16x2 r1095, %67, %97;
}
{
mul.f16x2 r1098, r1095, r993;
}
{
add.f16x2 r1101, %57, r1098;
}
{
add.f16x2 r1104, %77, %87;
}
{
mul.f16x2 r1107, r1104, r997;
}
{
add.f16x2 r1110, r1101, r1107;
}
{
sub.f16x2 r1113, %68, %98;
}
{
mul.f16x2 r1116, r1113, r995;
}
{
sub.f16x2 r1119, %78, %88;
}
{
mul.f16x2 r1122, r1119, r998;
}
{
add.f16x2 r1125, r1116, r1122;
}
{
sub.f16x2 r1128, r1110, r1125;
}
{
add.f16x2 r1131, %67, %97;
}
{
mul.f16x2 r1134, r1131, r993;
}
{
add.f16x2 r1137, %57, r1134;
}
{
add.f16x2 r1140, %77, %87;
}
{
mul.f16x2 r1143, r1140, r997;
}
{
add.f16x2 r1146, r1137, r1143;
}
{
sub.f16x2 r1149, %68, %98;
}
{
mul.f16x2 r1152, r1149, r995;
}
{
sub.f16x2 r1155, %78, %88;
}
{
mul.f16x2 r1158, r1155, r998;
}
{
add.f16x2 r1161, r1152, r1158;
}
{
add.f16x2 r1164, r1146, r1161;
}
{
add.f16x2 r1167, %68, %98;
}
{
mul.f16x2 r1170, r1167, r989;
}
{
add.f16x2 r1173, %58, r1170;
}
{
add.f16x2 r1176, %78, %88;
}
{
mul.f16x2 r1179, r1176, r993;
}
{
add.f16x2 r1182, r1173, r1179;
}
{
sub.f16x2 r1185, %67, %97;
}
{
mul.f16x2 r1188, r1185, r991;
}
{
sub.f16x2 r1191, %77, %87;
}
{
mul.f16x2 r1194, r1191, r995;
}
{
add.f16x2 r1197, r1188, r1194;
}
{
add.f16x2 r1200, r1182, r1197;
}
{
add.f16x2 r1203, %68, %98;
}
{
mul.f16x2 r1206, r1203, r989;
}
{
add.f16x2 r1209, %58, r1206;
}
{
add.f16x2 r1212, %78, %88;
}
{
mul.f16x2 r1215, r1212, r993;
}
{
add.f16x2 r1218, r1209, r1215;
}
{
sub.f16x2 r1221, %67, %97;
}
{
mul.f16x2 r1224, r1221, r991;
}
{
sub.f16x2 r1227, %77, %87;
}
{
mul.f16x2 r1230, r1227, r995;
}
{
add.f16x2 r1233, r1224, r1230;
}
{
sub.f16x2 r1236, r1218, r1233;
}
{
add.f16x2 r1239, %68, %98;
}
{
mul.f16x2 r1242, r1239, r993;
}
{
add.f16x2 r1245, %58, r1242;
}
{
add.f16x2 r1248, %78, %88;
}
{
mul.f16x2 r1251, r1248, r997;
}
{
add.f16x2 r1254, r1245, r1251;
}
{
sub.f16x2 r1257, %67, %97;
}
{
mul.f16x2 r1260, r1257, r995;
}
{
sub.f16x2 r1263, %77, %87;
}
{
mul.f16x2 r1266, r1263, r998;
}
{
add.f16x2 r1269, r1260, r1266;
}
{
add.f16x2 r1272, r1254, r1269;
}
{
add.f16x2 r1275, %68, %98;
}
{
mul.f16x2 r1278, r1275, r993;
}
{
add.f16x2 r1281, %58, r1278;
}
{
add.f16x2 r1284, %78, %88;
}
{
mul.f16x2 r1287, r1284, r997;
}
{
add.f16x2 r1290, r1281, r1287;
}
{
sub.f16x2 r1293, %67, %97;
}
{
mul.f16x2 r1296, r1293, r995;
}
{
sub.f16x2 r1299, %77, %87;
}
{
mul.f16x2 r1302, r1299, r998;
}
{
add.f16x2 r1305, r1296, r1302;
}
{
sub.f16x2 r1308, r1290, r1305;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1311, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1312, {low, high};
}
{
neg.f16x2 r1313, r1312;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1315, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1316, {low, high};
}
{
neg.f16x2 r1317, r1316;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1319, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1320, {low, high};
}
{
add.f16x2 r1321, %69, %99;
}
{
add.f16x2 r1324, %59, r1321;
}
{
add.f16x2 r1327, %79, %89;
}
{
add.f16x2 r1330, r1324, r1327;
}
{
add.f16x2 r1333, %70, %100;
}
{
add.f16x2 r1336, %60, r1333;
}
{
add.f16x2 r1339, %80, %90;
}
{
add.f16x2 r1342, r1336, r1339;
}
{
add.f16x2 r1345, %69, %99;
}
{
mul.f16x2 r1348, r1345, r1311;
}
{
add.f16x2 r1351, %59, r1348;
}
{
add.f16x2 r1354, %79, %89;
}
{
mul.f16x2 r1357, r1354, r1315;
}
{
add.f16x2 r1360, r1351, r1357;
}
{
sub.f16x2 r1363, %70, %100;
}
{
mul.f16x2 r1366, r1363, r1313;
}
{
sub.f16x2 r1369, %80, %90;
}
{
mul.f16x2 r1372, r1369, r1317;
}
{
add.f16x2 r1375, r1366, r1372;
}
{
sub.f16x2 r1378, r1360, r1375;
}
{
add.f16x2 r1381, %69, %99;
}
{
mul.f16x2 r1384, r1381, r1311;
}
{
add.f16x2 r1387, %59, r1384;
}
{
add.f16x2 r1390, %79, %89;
}
{
mul.f16x2 r1393, r1390, r1315;
}
{
add.f16x2 r1396, r1387, r1393;
}
{
sub.f16x2 r1399, %70, %100;
}
{
mul.f16x2 r1402, r1399, r1313;
}
{
sub.f16x2 r1405, %80, %90;
}
{
mul.f16x2 r1408, r1405, r1317;
}
{
add.f16x2 r1411, r1402, r1408;
}
{
add.f16x2 r1414, r1396, r1411;
}
{
add.f16x2 r1417, %69, %99;
}
{
mul.f16x2 r1420, r1417, r1315;
}
{
add.f16x2 r1423, %59, r1420;
}
{
add.f16x2 r1426, %79, %89;
}
{
mul.f16x2 r1429, r1426, r1319;
}
{
add.f16x2 r1432, r1423, r1429;
}
{
sub.f16x2 r1435, %70, %100;
}
{
mul.f16x2 r1438, r1435, r1317;
}
{
sub.f16x2 r1441, %80, %90;
}
{
mul.f16x2 r1444, r1441, r1320;
}
{
add.f16x2 r1447, r1438, r1444;
}
{
sub.f16x2 r1450, r1432, r1447;
}
{
add.f16x2 r1453, %69, %99;
}
{
mul.f16x2 r1456, r1453, r1315;
}
{
add.f16x2 r1459, %59, r1456;
}
{
add.f16x2 r1462, %79, %89;
}
{
mul.f16x2 r1465, r1462, r1319;
}
{
add.f16x2 r1468, r1459, r1465;
}
{
sub.f16x2 r1471, %70, %100;
}
{
mul.f16x2 r1474, r1471, r1317;
}
{
sub.f16x2 r1477, %80, %90;
}
{
mul.f16x2 r1480, r1477, r1320;
}
{
add.f16x2 r1483, r1474, r1480;
}
{
add.f16x2 r1486, r1468, r1483;
}
{
add.f16x2 r1489, %70, %100;
}
{
mul.f16x2 r1492, r1489, r1311;
}
{
add.f16x2 r1495, %60, r1492;
}
{
add.f16x2 r1498, %80, %90;
}
{
mul.f16x2 r1501, r1498, r1315;
}
{
add.f16x2 r1504, r1495, r1501;
}
{
sub.f16x2 r1507, %69, %99;
}
{
mul.f16x2 r1510, r1507, r1313;
}
{
sub.f16x2 r1513, %79, %89;
}
{
mul.f16x2 r1516, r1513, r1317;
}
{
add.f16x2 r1519, r1510, r1516;
}
{
add.f16x2 r1522, r1504, r1519;
}
{
add.f16x2 r1525, %70, %100;
}
{
mul.f16x2 r1528, r1525, r1311;
}
{
add.f16x2 r1531, %60, r1528;
}
{
add.f16x2 r1534, %80, %90;
}
{
mul.f16x2 r1537, r1534, r1315;
}
{
add.f16x2 r1540, r1531, r1537;
}
{
sub.f16x2 r1543, %69, %99;
}
{
mul.f16x2 r1546, r1543, r1313;
}
{
sub.f16x2 r1549, %79, %89;
}
{
mul.f16x2 r1552, r1549, r1317;
}
{
add.f16x2 r1555, r1546, r1552;
}
{
sub.f16x2 r1558, r1540, r1555;
}
{
add.f16x2 r1561, %70, %100;
}
{
mul.f16x2 r1564, r1561, r1315;
}
{
add.f16x2 r1567, %60, r1564;
}
{
add.f16x2 r1570, %80, %90;
}
{
mul.f16x2 r1573, r1570, r1319;
}
{
add.f16x2 r1576, r1567, r1573;
}
{
sub.f16x2 r1579, %69, %99;
}
{
mul.f16x2 r1582, r1579, r1317;
}
{
sub.f16x2 r1585, %79, %89;
}
{
mul.f16x2 r1588, r1585, r1320;
}
{
add.f16x2 r1591, r1582, r1588;
}
{
add.f16x2 r1594, r1576, r1591;
}
{
add.f16x2 r1597, %70, %100;
}
{
mul.f16x2 r1600, r1597, r1315;
}
{
add.f16x2 r1603, %60, r1600;
}
{
add.f16x2 r1606, %80, %90;
}
{
mul.f16x2 r1609, r1606, r1319;
}
{
add.f16x2 r1612, r1603, r1609;
}
{
sub.f16x2 r1615, %69, %99;
}
{
mul.f16x2 r1618, r1615, r1317;
}
{
sub.f16x2 r1621, %79, %89;
}
{
mul.f16x2 r1624, r1621, r1320;
}
{
add.f16x2 r1627, r1618, r1624;
}
{
sub.f16x2 r1630, r1612, r1627;
}
mov.f32 f62, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r1633, {low, high};
}
mov.f32 f64, 0f3E7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r1634, {low, high};
}
mov.f32 f66, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r1635, {low, high};
}
mov.f32 f68, 0f3EF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r1636, {low, high};
}
mov.f32 f70, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r1637, {low, high};
}
mov.f32 f72, 0f3F2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r1638, {low, high};
}
mov.f32 f74, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r1639, {low, high};
}
mov.f32 f76, 0f3F5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r1640, {low, high};
}
mov.f32 f82, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1643, {low, high};
}
mov.f32 f84, 0f3F7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1644, {low, high};
}
mov.f32 f90, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r1647, {low, high};
}
mov.f32 f92, 0f3F67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r1648, {low, high};
}
mov.f32 f122, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1649, {low, high};
}
mov.f32 f96, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r1650, {low, high};
}
mov.f32 f106, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1655, {low, high};
}
mov.f32 f108, 0f3E00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1656, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r1663, {low, high};
}
mov.f32 f124, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r1664, {low, high};
}
{
mul.f16x2 r1681, r412, r1633;
}
{
mul.f16x2 r1684, r556, r1634;
}
{
sub.f16x2 r1687, r1681, r1684;
}
{
mul.f16x2 r1690, r412, r1634;
}
{
fma.rn.f16x2 r1693, r556, r1633, r1690;
}
{
mul.f16x2 r1697, r734, r1635;
}
{
mul.f16x2 r1700, r878, r1636;
}
{
sub.f16x2 r1703, r1697, r1700;
}
{
mul.f16x2 r1706, r734, r1636;
}
{
fma.rn.f16x2 r1709, r878, r1635, r1706;
}
{
mul.f16x2 r1713, r1056, r1637;
}
{
mul.f16x2 r1716, r1200, r1638;
}
{
sub.f16x2 r1719, r1713, r1716;
}
{
mul.f16x2 r1722, r1056, r1638;
}
{
fma.rn.f16x2 r1725, r1200, r1637, r1722;
}
{
mul.f16x2 r1729, r1378, r1639;
}
{
mul.f16x2 r1732, r1522, r1640;
}
{
sub.f16x2 r1735, r1729, r1732;
}
{
mul.f16x2 r1738, r1378, r1640;
}
{
fma.rn.f16x2 r1741, r1522, r1639, r1738;
}
{
mul.f16x2 r1745, r484, r1635;
}
{
mul.f16x2 r1748, r628, r1636;
}
{
sub.f16x2 r1751, r1745, r1748;
}
{
mul.f16x2 r1754, r484, r1636;
}
{
fma.rn.f16x2 r1757, r628, r1635, r1754;
}
{
mul.f16x2 r1761, r806, r1639;
}
{
mul.f16x2 r1764, r950, r1640;
}
{
sub.f16x2 r1767, r1761, r1764;
}
{
mul.f16x2 r1770, r806, r1640;
}
{
fma.rn.f16x2 r1773, r950, r1639, r1770;
}
{
mul.f16x2 r1777, r1128, r1643;
}
{
mul.f16x2 r1780, r1272, r1644;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r1128, r1644;
}
{
fma.rn.f16x2 r1789, r1272, r1643, r1786;
}
{
mul.f16x2 r1793, r1450, r1647;
}
{
mul.f16x2 r1796, r1594, r1648;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r1450, r1648;
}
{
fma.rn.f16x2 r1805, r1594, r1647, r1802;
}
{
mul.f16x2 r1809, r520, r1637;
}
{
mul.f16x2 r1812, r664, r1638;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r520, r1638;
}
{
fma.rn.f16x2 r1821, r664, r1637, r1818;
}
{
mul.f16x2 r1825, r842, r1643;
}
{
mul.f16x2 r1828, r986, r1644;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r842, r1644;
}
{
fma.rn.f16x2 r1837, r986, r1643, r1834;
}
{
mul.f16x2 r1841, r1164, r1649;
}
{
mul.f16x2 r1844, r1308, r1650;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r1164, r1650;
}
{
fma.rn.f16x2 r1853, r1308, r1649, r1850;
}
{
mul.f16x2 r1857, r1486, r1655;
}
{
mul.f16x2 r1860, r1630, r1656;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r1486, r1656;
}
{
fma.rn.f16x2 r1869, r1630, r1655, r1866;
}
{
mul.f16x2 r1873, r448, r1639;
}
{
mul.f16x2 r1876, r592, r1640;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r448, r1640;
}
{
fma.rn.f16x2 r1885, r592, r1639, r1882;
}
{
mul.f16x2 r1889, r770, r1647;
}
{
mul.f16x2 r1892, r914, r1648;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r770, r1648;
}
{
fma.rn.f16x2 r1901, r914, r1647, r1898;
}
{
mul.f16x2 r1905, r1092, r1655;
}
{
mul.f16x2 r1908, r1236, r1656;
}
{
sub.f16x2 r1911, r1905, r1908;
}
{
mul.f16x2 r1914, r1092, r1656;
}
{
fma.rn.f16x2 r1917, r1236, r1655, r1914;
}
{
mul.f16x2 r1921, r1414, r1663;
}
{
mul.f16x2 r1924, r1558, r1664;
}
{
sub.f16x2 r1927, r1921, r1924;
}
{
mul.f16x2 r1930, r1414, r1664;
}
{
fma.rn.f16x2 r1933, r1558, r1663, r1930;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1937, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1938, {low, high};
}
{
neg.f16x2 r1939, r1938;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r1941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r1942, {low, high};
}
{
neg.f16x2 r1943, r1942;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r1945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r1946, {low, high};
}
{
add.f16x2 r1947, r364, r1330;
}
{
add.f16x2 r1950, r42, r1947;
}
{
add.f16x2 r1953, r686, r1008;
}
{
add.f16x2 r1956, r1950, r1953;
}
st.local.u32 [rd3], r1956;
{
add.f16x2 r1959, r376, r1342;
}
{
add.f16x2 r1962, r54, r1959;
}
{
add.f16x2 r1965, r698, r1020;
}
{
add.f16x2 r1968, r1962, r1965;
}
st.local.u32 [rd3+4], r1968;
{
add.f16x2 r1971, r364, r1330;
}
{
mul.f16x2 r1974, r1971, r1937;
}
{
add.f16x2 r1977, r42, r1974;
}
{
add.f16x2 r1980, r686, r1008;
}
{
mul.f16x2 r1983, r1980, r1941;
}
{
add.f16x2 r1986, r1977, r1983;
}
{
sub.f16x2 r1989, r376, r1342;
}
{
mul.f16x2 r1992, r1989, r1939;
}
{
sub.f16x2 r1995, r698, r1020;
}
{
mul.f16x2 r1998, r1995, r1943;
}
{
add.f16x2 r2001, r1992, r1998;
}
{
sub.f16x2 r2004, r1986, r2001;
}
st.local.u32 [rd3+40], r2004;
{
add.f16x2 r2007, r364, r1330;
}
{
mul.f16x2 r2010, r2007, r1937;
}
{
add.f16x2 r2013, r42, r2010;
}
{
add.f16x2 r2016, r686, r1008;
}
{
mul.f16x2 r2019, r2016, r1941;
}
{
add.f16x2 r2022, r2013, r2019;
}
{
sub.f16x2 r2025, r376, r1342;
}
{
mul.f16x2 r2028, r2025, r1939;
}
{
sub.f16x2 r2031, r698, r1020;
}
{
mul.f16x2 r2034, r2031, r1943;
}
{
add.f16x2 r2037, r2028, r2034;
}
{
add.f16x2 r2040, r2022, r2037;
}
st.local.u32 [rd3+160], r2040;
{
add.f16x2 r2043, r364, r1330;
}
{
mul.f16x2 r2046, r2043, r1941;
}
{
add.f16x2 r2049, r42, r2046;
}
{
add.f16x2 r2052, r686, r1008;
}
{
mul.f16x2 r2055, r2052, r1945;
}
{
add.f16x2 r2058, r2049, r2055;
}
{
sub.f16x2 r2061, r376, r1342;
}
{
mul.f16x2 r2064, r2061, r1943;
}
{
sub.f16x2 r2067, r698, r1020;
}
{
mul.f16x2 r2070, r2067, r1946;
}
{
add.f16x2 r2073, r2064, r2070;
}
{
sub.f16x2 r2076, r2058, r2073;
}
st.local.u32 [rd3+80], r2076;
{
add.f16x2 r2079, r364, r1330;
}
{
mul.f16x2 r2082, r2079, r1941;
}
{
add.f16x2 r2085, r42, r2082;
}
{
add.f16x2 r2088, r686, r1008;
}
{
mul.f16x2 r2091, r2088, r1945;
}
{
add.f16x2 r2094, r2085, r2091;
}
{
sub.f16x2 r2097, r376, r1342;
}
{
mul.f16x2 r2100, r2097, r1943;
}
{
sub.f16x2 r2103, r698, r1020;
}
{
mul.f16x2 r2106, r2103, r1946;
}
{
add.f16x2 r2109, r2100, r2106;
}
{
add.f16x2 r2112, r2094, r2109;
}
st.local.u32 [rd3+120], r2112;
{
add.f16x2 r2115, r376, r1342;
}
{
mul.f16x2 r2118, r2115, r1937;
}
{
add.f16x2 r2121, r54, r2118;
}
{
add.f16x2 r2124, r698, r1020;
}
{
mul.f16x2 r2127, r2124, r1941;
}
{
add.f16x2 r2130, r2121, r2127;
}
{
sub.f16x2 r2133, r364, r1330;
}
{
mul.f16x2 r2136, r2133, r1939;
}
{
sub.f16x2 r2139, r686, r1008;
}
{
mul.f16x2 r2142, r2139, r1943;
}
{
add.f16x2 r2145, r2136, r2142;
}
{
add.f16x2 r2148, r2130, r2145;
}
st.local.u32 [rd3+44], r2148;
{
add.f16x2 r2151, r376, r1342;
}
{
mul.f16x2 r2154, r2151, r1937;
}
{
add.f16x2 r2157, r54, r2154;
}
{
add.f16x2 r2160, r698, r1020;
}
{
mul.f16x2 r2163, r2160, r1941;
}
{
add.f16x2 r2166, r2157, r2163;
}
{
sub.f16x2 r2169, r364, r1330;
}
{
mul.f16x2 r2172, r2169, r1939;
}
{
sub.f16x2 r2175, r686, r1008;
}
{
mul.f16x2 r2178, r2175, r1943;
}
{
add.f16x2 r2181, r2172, r2178;
}
{
sub.f16x2 r2184, r2166, r2181;
}
st.local.u32 [rd3+164], r2184;
{
add.f16x2 r2187, r376, r1342;
}
{
mul.f16x2 r2190, r2187, r1941;
}
{
add.f16x2 r2193, r54, r2190;
}
{
add.f16x2 r2196, r698, r1020;
}
{
mul.f16x2 r2199, r2196, r1945;
}
{
add.f16x2 r2202, r2193, r2199;
}
{
sub.f16x2 r2205, r364, r1330;
}
{
mul.f16x2 r2208, r2205, r1943;
}
{
sub.f16x2 r2211, r686, r1008;
}
{
mul.f16x2 r2214, r2211, r1946;
}
{
add.f16x2 r2217, r2208, r2214;
}
{
add.f16x2 r2220, r2202, r2217;
}
st.local.u32 [rd3+84], r2220;
{
add.f16x2 r2223, r376, r1342;
}
{
mul.f16x2 r2226, r2223, r1941;
}
{
add.f16x2 r2229, r54, r2226;
}
{
add.f16x2 r2232, r698, r1020;
}
{
mul.f16x2 r2235, r2232, r1945;
}
{
add.f16x2 r2238, r2229, r2235;
}
{
sub.f16x2 r2241, r364, r1330;
}
{
mul.f16x2 r2244, r2241, r1943;
}
{
sub.f16x2 r2247, r686, r1008;
}
{
mul.f16x2 r2250, r2247, r1946;
}
{
add.f16x2 r2253, r2244, r2250;
}
{
sub.f16x2 r2256, r2238, r2253;
}
st.local.u32 [rd3+124], r2256;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2259, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2260, {low, high};
}
{
neg.f16x2 r2261, r2260;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2264, {low, high};
}
{
neg.f16x2 r2265, r2264;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2267, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2268, {low, high};
}
{
add.f16x2 r2269, r1687, r1735;
}
{
add.f16x2 r2272, r90, r2269;
}
{
add.f16x2 r2275, r1703, r1719;
}
{
add.f16x2 r11308, r2272, r2275;
}
st.local.u32 [rd3+8], r11308;
{
add.f16x2 r2281, r1693, r1741;
}
{
add.f16x2 r2284, r234, r2281;
}
{
add.f16x2 r2287, r1709, r1725;
}
{
add.f16x2 r2290, r2284, r2287;
}
st.local.u32 [rd3+12], r2290;
{
add.f16x2 r2293, r1687, r1735;
}
{
mul.f16x2 r2296, r2293, r2259;
}
{
add.f16x2 r2299, r90, r2296;
}
{
add.f16x2 r2302, r1703, r1719;
}
{
mul.f16x2 r2305, r2302, r2263;
}
{
add.f16x2 r2308, r2299, r2305;
}
{
sub.f16x2 r2311, r1693, r1741;
}
{
mul.f16x2 r2314, r2311, r2261;
}
{
sub.f16x2 r2317, r1709, r1725;
}
{
mul.f16x2 r2320, r2317, r2265;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
sub.f16x2 r2326, r2308, r2323;
}
st.local.u32 [rd3+48], r2326;
{
add.f16x2 r2329, r1687, r1735;
}
{
mul.f16x2 r2332, r2329, r2259;
}
{
add.f16x2 r2335, r90, r2332;
}
{
add.f16x2 r2338, r1703, r1719;
}
{
mul.f16x2 r2341, r2338, r2263;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
sub.f16x2 r2347, r1693, r1741;
}
{
mul.f16x2 r2350, r2347, r2261;
}
{
sub.f16x2 r2353, r1709, r1725;
}
{
mul.f16x2 r2356, r2353, r2265;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
add.f16x2 r2362, r2344, r2359;
}
st.local.u32 [rd3+168], r2362;
{
add.f16x2 r2365, r1687, r1735;
}
{
mul.f16x2 r2368, r2365, r2263;
}
{
add.f16x2 r2371, r90, r2368;
}
{
add.f16x2 r2374, r1703, r1719;
}
{
mul.f16x2 r2377, r2374, r2267;
}
{
add.f16x2 r2380, r2371, r2377;
}
{
sub.f16x2 r2383, r1693, r1741;
}
{
mul.f16x2 r2386, r2383, r2265;
}
{
sub.f16x2 r2389, r1709, r1725;
}
{
mul.f16x2 r2392, r2389, r2268;
}
{
add.f16x2 r2395, r2386, r2392;
}
{
sub.f16x2 r2398, r2380, r2395;
}
st.local.u32 [rd3+88], r2398;
{
add.f16x2 r2401, r1687, r1735;
}
{
mul.f16x2 r2404, r2401, r2263;
}
{
add.f16x2 r2407, r90, r2404;
}
{
add.f16x2 r2410, r1703, r1719;
}
{
mul.f16x2 r2413, r2410, r2267;
}
{
add.f16x2 r2416, r2407, r2413;
}
{
sub.f16x2 r2419, r1693, r1741;
}
{
mul.f16x2 r2422, r2419, r2265;
}
{
sub.f16x2 r2425, r1709, r1725;
}
{
mul.f16x2 r2428, r2425, r2268;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
add.f16x2 r2434, r2416, r2431;
}
st.local.u32 [rd3+128], r2434;
{
add.f16x2 r2437, r1693, r1741;
}
{
mul.f16x2 r2440, r2437, r2259;
}
{
add.f16x2 r2443, r234, r2440;
}
{
add.f16x2 r2446, r1709, r1725;
}
{
mul.f16x2 r2449, r2446, r2263;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
sub.f16x2 r2455, r1687, r1735;
}
{
mul.f16x2 r2458, r2455, r2261;
}
{
sub.f16x2 r2461, r1703, r1719;
}
{
mul.f16x2 r2464, r2461, r2265;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
add.f16x2 r2470, r2452, r2467;
}
st.local.u32 [rd3+52], r2470;
{
add.f16x2 r2473, r1693, r1741;
}
{
mul.f16x2 r2476, r2473, r2259;
}
{
add.f16x2 r2479, r234, r2476;
}
{
add.f16x2 r2482, r1709, r1725;
}
{
mul.f16x2 r2485, r2482, r2263;
}
{
add.f16x2 r2488, r2479, r2485;
}
{
sub.f16x2 r2491, r1687, r1735;
}
{
mul.f16x2 r2494, r2491, r2261;
}
{
sub.f16x2 r2497, r1703, r1719;
}
{
mul.f16x2 r2500, r2497, r2265;
}
{
add.f16x2 r2503, r2494, r2500;
}
{
sub.f16x2 r2506, r2488, r2503;
}
st.local.u32 [rd3+172], r2506;
{
add.f16x2 r2509, r1693, r1741;
}
{
mul.f16x2 r2512, r2509, r2263;
}
{
add.f16x2 r2515, r234, r2512;
}
{
add.f16x2 r2518, r1709, r1725;
}
{
mul.f16x2 r2521, r2518, r2267;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
sub.f16x2 r2527, r1687, r1735;
}
{
mul.f16x2 r2530, r2527, r2265;
}
{
sub.f16x2 r2533, r1703, r1719;
}
{
mul.f16x2 r2536, r2533, r2268;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
add.f16x2 r2542, r2524, r2539;
}
st.local.u32 [rd3+92], r2542;
{
add.f16x2 r2545, r1693, r1741;
}
{
mul.f16x2 r2548, r2545, r2263;
}
{
add.f16x2 r2551, r234, r2548;
}
{
add.f16x2 r2554, r1709, r1725;
}
{
mul.f16x2 r2557, r2554, r2267;
}
{
add.f16x2 r2560, r2551, r2557;
}
{
sub.f16x2 r2563, r1687, r1735;
}
{
mul.f16x2 r2566, r2563, r2265;
}
{
sub.f16x2 r2569, r1703, r1719;
}
{
mul.f16x2 r2572, r2569, r2268;
}
{
add.f16x2 r2575, r2566, r2572;
}
{
sub.f16x2 r2578, r2560, r2575;
}
st.local.u32 [rd3+132], r2578;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2582, {low, high};
}
{
neg.f16x2 r2583, r2582;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2585, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2586, {low, high};
}
{
neg.f16x2 r2587, r2586;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2589, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2590, {low, high};
}
{
add.f16x2 r2591, r1751, r1799;
}
{
add.f16x2 r2594, r162, r2591;
}
{
add.f16x2 r2597, r1767, r1783;
}
{
add.f16x2 r2600, r2594, r2597;
}
st.local.u32 [rd3+16], r2600;
{
add.f16x2 r2603, r1757, r1805;
}
{
add.f16x2 r2606, r306, r2603;
}
{
add.f16x2 r2609, r1773, r1789;
}
{
add.f16x2 r2612, r2606, r2609;
}
st.local.u32 [rd3+20], r2612;
{
add.f16x2 r2615, r1751, r1799;
}
{
mul.f16x2 r2618, r2615, r2581;
}
{
add.f16x2 r2621, r162, r2618;
}
{
add.f16x2 r2624, r1767, r1783;
}
{
mul.f16x2 r2627, r2624, r2585;
}
{
add.f16x2 r2630, r2621, r2627;
}
{
sub.f16x2 r2633, r1757, r1805;
}
{
mul.f16x2 r2636, r2633, r2583;
}
{
sub.f16x2 r2639, r1773, r1789;
}
{
mul.f16x2 r2642, r2639, r2587;
}
{
add.f16x2 r2645, r2636, r2642;
}
{
sub.f16x2 r2648, r2630, r2645;
}
st.local.u32 [rd3+56], r2648;
{
add.f16x2 r2651, r1751, r1799;
}
{
mul.f16x2 r2654, r2651, r2581;
}
{
add.f16x2 r2657, r162, r2654;
}
{
add.f16x2 r2660, r1767, r1783;
}
{
mul.f16x2 r2663, r2660, r2585;
}
{
add.f16x2 r2666, r2657, r2663;
}
{
sub.f16x2 r2669, r1757, r1805;
}
{
mul.f16x2 r2672, r2669, r2583;
}
{
sub.f16x2 r2675, r1773, r1789;
}
{
mul.f16x2 r2678, r2675, r2587;
}
{
add.f16x2 r2681, r2672, r2678;
}
{
add.f16x2 r2684, r2666, r2681;
}
st.local.u32 [rd3+176], r2684;
{
add.f16x2 r2687, r1751, r1799;
}
{
mul.f16x2 r2690, r2687, r2585;
}
{
add.f16x2 r2693, r162, r2690;
}
{
add.f16x2 r2696, r1767, r1783;
}
{
mul.f16x2 r2699, r2696, r2589;
}
{
add.f16x2 r2702, r2693, r2699;
}
{
sub.f16x2 r2705, r1757, r1805;
}
{
mul.f16x2 r2708, r2705, r2587;
}
{
sub.f16x2 r2711, r1773, r1789;
}
{
mul.f16x2 r2714, r2711, r2590;
}
{
add.f16x2 r2717, r2708, r2714;
}
{
sub.f16x2 r2720, r2702, r2717;
}
st.local.u32 [rd3+96], r2720;
{
add.f16x2 r2723, r1751, r1799;
}
{
mul.f16x2 r2726, r2723, r2585;
}
{
add.f16x2 r2729, r162, r2726;
}
{
add.f16x2 r2732, r1767, r1783;
}
{
mul.f16x2 r2735, r2732, r2589;
}
{
add.f16x2 r2738, r2729, r2735;
}
{
sub.f16x2 r2741, r1757, r1805;
}
{
mul.f16x2 r2744, r2741, r2587;
}
{
sub.f16x2 r2747, r1773, r1789;
}
{
mul.f16x2 r2750, r2747, r2590;
}
{
add.f16x2 r2753, r2744, r2750;
}
{
add.f16x2 r2756, r2738, r2753;
}
st.local.u32 [rd3+136], r2756;
{
add.f16x2 r2759, r1757, r1805;
}
{
mul.f16x2 r2762, r2759, r2581;
}
{
add.f16x2 r2765, r306, r2762;
}
{
add.f16x2 r2768, r1773, r1789;
}
{
mul.f16x2 r2771, r2768, r2585;
}
{
add.f16x2 r2774, r2765, r2771;
}
{
sub.f16x2 r2777, r1751, r1799;
}
{
mul.f16x2 r2780, r2777, r2583;
}
{
sub.f16x2 r2783, r1767, r1783;
}
{
mul.f16x2 r2786, r2783, r2587;
}
{
add.f16x2 r2789, r2780, r2786;
}
{
add.f16x2 r2792, r2774, r2789;
}
st.local.u32 [rd3+60], r2792;
{
add.f16x2 r2795, r1757, r1805;
}
{
mul.f16x2 r2798, r2795, r2581;
}
{
add.f16x2 r2801, r306, r2798;
}
{
add.f16x2 r2804, r1773, r1789;
}
{
mul.f16x2 r2807, r2804, r2585;
}
{
add.f16x2 r2810, r2801, r2807;
}
{
sub.f16x2 r2813, r1751, r1799;
}
{
mul.f16x2 r2816, r2813, r2583;
}
{
sub.f16x2 r2819, r1767, r1783;
}
{
mul.f16x2 r2822, r2819, r2587;
}
{
add.f16x2 r2825, r2816, r2822;
}
{
sub.f16x2 r2828, r2810, r2825;
}
st.local.u32 [rd3+180], r2828;
{
add.f16x2 r2831, r1757, r1805;
}
{
mul.f16x2 r2834, r2831, r2585;
}
{
add.f16x2 r2837, r306, r2834;
}
{
add.f16x2 r2840, r1773, r1789;
}
{
mul.f16x2 r2843, r2840, r2589;
}
{
add.f16x2 r2846, r2837, r2843;
}
{
sub.f16x2 r2849, r1751, r1799;
}
{
mul.f16x2 r2852, r2849, r2587;
}
{
sub.f16x2 r2855, r1767, r1783;
}
{
mul.f16x2 r2858, r2855, r2590;
}
{
add.f16x2 r2861, r2852, r2858;
}
{
add.f16x2 r2864, r2846, r2861;
}
st.local.u32 [rd3+100], r2864;
{
add.f16x2 r2867, r1757, r1805;
}
{
mul.f16x2 r2870, r2867, r2585;
}
{
add.f16x2 r2873, r306, r2870;
}
{
add.f16x2 r2876, r1773, r1789;
}
{
mul.f16x2 r2879, r2876, r2589;
}
{
add.f16x2 r2882, r2873, r2879;
}
{
sub.f16x2 r2885, r1751, r1799;
}
{
mul.f16x2 r2888, r2885, r2587;
}
{
sub.f16x2 r2891, r1767, r1783;
}
{
mul.f16x2 r2894, r2891, r2590;
}
{
add.f16x2 r2897, r2888, r2894;
}
{
sub.f16x2 r2900, r2882, r2897;
}
st.local.u32 [rd3+140], r2900;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2903, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2904, {low, high};
}
{
neg.f16x2 r2905, r2904;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r2907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r2908, {low, high};
}
{
neg.f16x2 r2909, r2908;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r2911, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r2912, {low, high};
}
{
add.f16x2 r2913, r1815, r1863;
}
{
add.f16x2 r2916, r198, r2913;
}
{
add.f16x2 r2919, r1831, r1847;
}
{
add.f16x2 r2922, r2916, r2919;
}
st.local.u32 [rd3+24], r2922;
{
add.f16x2 r2925, r1821, r1869;
}
{
add.f16x2 r2928, r342, r2925;
}
{
add.f16x2 r2931, r1837, r1853;
}
{
add.f16x2 r2934, r2928, r2931;
}
st.local.u32 [rd3+28], r2934;
{
add.f16x2 r2937, r1815, r1863;
}
{
mul.f16x2 r2940, r2937, r2903;
}
{
add.f16x2 r2943, r198, r2940;
}
{
add.f16x2 r2946, r1831, r1847;
}
{
mul.f16x2 r2949, r2946, r2907;
}
{
add.f16x2 r2952, r2943, r2949;
}
{
sub.f16x2 r2955, r1821, r1869;
}
{
mul.f16x2 r2958, r2955, r2905;
}
{
sub.f16x2 r2961, r1837, r1853;
}
{
mul.f16x2 r2964, r2961, r2909;
}
{
add.f16x2 r2967, r2958, r2964;
}
{
sub.f16x2 r2970, r2952, r2967;
}
st.local.u32 [rd3+64], r2970;
{
add.f16x2 r2973, r1815, r1863;
}
{
mul.f16x2 r2976, r2973, r2903;
}
{
add.f16x2 r2979, r198, r2976;
}
{
add.f16x2 r2982, r1831, r1847;
}
{
mul.f16x2 r2985, r2982, r2907;
}
{
add.f16x2 r2988, r2979, r2985;
}
{
sub.f16x2 r2991, r1821, r1869;
}
{
mul.f16x2 r2994, r2991, r2905;
}
{
sub.f16x2 r2997, r1837, r1853;
}
{
mul.f16x2 r3000, r2997, r2909;
}
{
add.f16x2 r3003, r2994, r3000;
}
{
add.f16x2 r3006, r2988, r3003;
}
st.local.u32 [rd3+184], r3006;
{
add.f16x2 r3009, r1815, r1863;
}
{
mul.f16x2 r3012, r3009, r2907;
}
{
add.f16x2 r3015, r198, r3012;
}
{
add.f16x2 r3018, r1831, r1847;
}
{
mul.f16x2 r3021, r3018, r2911;
}
{
add.f16x2 r3024, r3015, r3021;
}
{
sub.f16x2 r3027, r1821, r1869;
}
{
mul.f16x2 r3030, r3027, r2909;
}
{
sub.f16x2 r3033, r1837, r1853;
}
{
mul.f16x2 r3036, r3033, r2912;
}
{
add.f16x2 r3039, r3030, r3036;
}
{
sub.f16x2 r3042, r3024, r3039;
}
st.local.u32 [rd3+104], r3042;
{
add.f16x2 r3045, r1815, r1863;
}
{
mul.f16x2 r3048, r3045, r2907;
}
{
add.f16x2 r3051, r198, r3048;
}
{
add.f16x2 r3054, r1831, r1847;
}
{
mul.f16x2 r3057, r3054, r2911;
}
{
add.f16x2 r3060, r3051, r3057;
}
{
sub.f16x2 r3063, r1821, r1869;
}
{
mul.f16x2 r3066, r3063, r2909;
}
{
sub.f16x2 r3069, r1837, r1853;
}
{
mul.f16x2 r3072, r3069, r2912;
}
{
add.f16x2 r3075, r3066, r3072;
}
{
add.f16x2 r3078, r3060, r3075;
}
st.local.u32 [rd3+144], r3078;
{
add.f16x2 r3081, r1821, r1869;
}
{
mul.f16x2 r3084, r3081, r2903;
}
{
add.f16x2 r3087, r342, r3084;
}
{
add.f16x2 r3090, r1837, r1853;
}
{
mul.f16x2 r3093, r3090, r2907;
}
{
add.f16x2 r3096, r3087, r3093;
}
{
sub.f16x2 r3099, r1815, r1863;
}
{
mul.f16x2 r3102, r3099, r2905;
}
{
sub.f16x2 r3105, r1831, r1847;
}
{
mul.f16x2 r3108, r3105, r2909;
}
{
add.f16x2 r3111, r3102, r3108;
}
{
add.f16x2 r3114, r3096, r3111;
}
st.local.u32 [rd3+68], r3114;
{
add.f16x2 r3117, r1821, r1869;
}
{
mul.f16x2 r3120, r3117, r2903;
}
{
add.f16x2 r3123, r342, r3120;
}
{
add.f16x2 r3126, r1837, r1853;
}
{
mul.f16x2 r3129, r3126, r2907;
}
{
add.f16x2 r3132, r3123, r3129;
}
{
sub.f16x2 r3135, r1815, r1863;
}
{
mul.f16x2 r3138, r3135, r2905;
}
{
sub.f16x2 r3141, r1831, r1847;
}
{
mul.f16x2 r3144, r3141, r2909;
}
{
add.f16x2 r3147, r3138, r3144;
}
{
sub.f16x2 r3150, r3132, r3147;
}
st.local.u32 [rd3+188], r3150;
{
add.f16x2 r3153, r1821, r1869;
}
{
mul.f16x2 r3156, r3153, r2907;
}
{
add.f16x2 r3159, r342, r3156;
}
{
add.f16x2 r3162, r1837, r1853;
}
{
mul.f16x2 r3165, r3162, r2911;
}
{
add.f16x2 r3168, r3159, r3165;
}
{
sub.f16x2 r3171, r1815, r1863;
}
{
mul.f16x2 r3174, r3171, r2909;
}
{
sub.f16x2 r3177, r1831, r1847;
}
{
mul.f16x2 r3180, r3177, r2912;
}
{
add.f16x2 r3183, r3174, r3180;
}
{
add.f16x2 r3186, r3168, r3183;
}
st.local.u32 [rd3+108], r3186;
{
add.f16x2 r3189, r1821, r1869;
}
{
mul.f16x2 r3192, r3189, r2907;
}
{
add.f16x2 r3195, r342, r3192;
}
{
add.f16x2 r3198, r1837, r1853;
}
{
mul.f16x2 r3201, r3198, r2911;
}
{
add.f16x2 r3204, r3195, r3201;
}
{
sub.f16x2 r3207, r1815, r1863;
}
{
mul.f16x2 r3210, r3207, r2909;
}
{
sub.f16x2 r3213, r1831, r1847;
}
{
mul.f16x2 r3216, r3213, r2912;
}
{
add.f16x2 r3219, r3210, r3216;
}
{
sub.f16x2 r3222, r3204, r3219;
}
st.local.u32 [rd3+148], r3222;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3226, {low, high};
}
{
neg.f16x2 r3227, r3226;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3230, {low, high};
}
{
neg.f16x2 r3231, r3230;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3233, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3234, {low, high};
}
{
add.f16x2 r3235, r1879, r1927;
}
{
add.f16x2 r3238, r126, r3235;
}
{
add.f16x2 r3241, r1895, r1911;
}
{
add.f16x2 r3244, r3238, r3241;
}
st.local.u32 [rd3+32], r3244;
{
add.f16x2 r3247, r1885, r1933;
}
{
add.f16x2 r3250, r270, r3247;
}
{
add.f16x2 r3253, r1901, r1917;
}
{
add.f16x2 r3256, r3250, r3253;
}
st.local.u32 [rd3+36], r3256;
{
add.f16x2 r3259, r1879, r1927;
}
{
mul.f16x2 r3262, r3259, r3225;
}
{
add.f16x2 r3265, r126, r3262;
}
{
add.f16x2 r3268, r1895, r1911;
}
{
mul.f16x2 r3271, r3268, r3229;
}
{
add.f16x2 r3274, r3265, r3271;
}
{
sub.f16x2 r3277, r1885, r1933;
}
{
mul.f16x2 r3280, r3277, r3227;
}
{
sub.f16x2 r3283, r1901, r1917;
}
{
mul.f16x2 r3286, r3283, r3231;
}
{
add.f16x2 r3289, r3280, r3286;
}
{
sub.f16x2 r3292, r3274, r3289;
}
st.local.u32 [rd3+72], r3292;
{
add.f16x2 r3295, r1879, r1927;
}
{
mul.f16x2 r3298, r3295, r3225;
}
{
add.f16x2 r3301, r126, r3298;
}
{
add.f16x2 r3304, r1895, r1911;
}
{
mul.f16x2 r3307, r3304, r3229;
}
{
add.f16x2 r3310, r3301, r3307;
}
{
sub.f16x2 r3313, r1885, r1933;
}
{
mul.f16x2 r3316, r3313, r3227;
}
{
sub.f16x2 r3319, r1901, r1917;
}
{
mul.f16x2 r3322, r3319, r3231;
}
{
add.f16x2 r3325, r3316, r3322;
}
{
add.f16x2 r3328, r3310, r3325;
}
st.local.u32 [rd3+192], r3328;
{
add.f16x2 r3331, r1879, r1927;
}
{
mul.f16x2 r3334, r3331, r3229;
}
{
add.f16x2 r3337, r126, r3334;
}
{
add.f16x2 r3340, r1895, r1911;
}
{
mul.f16x2 r3343, r3340, r3233;
}
{
add.f16x2 r3346, r3337, r3343;
}
{
sub.f16x2 r3349, r1885, r1933;
}
{
mul.f16x2 r3352, r3349, r3231;
}
{
sub.f16x2 r3355, r1901, r1917;
}
{
mul.f16x2 r3358, r3355, r3234;
}
{
add.f16x2 r3361, r3352, r3358;
}
{
sub.f16x2 r3364, r3346, r3361;
}
st.local.u32 [rd3+112], r3364;
{
add.f16x2 r3367, r1879, r1927;
}
{
mul.f16x2 r3370, r3367, r3229;
}
{
add.f16x2 r3373, r126, r3370;
}
{
add.f16x2 r3376, r1895, r1911;
}
{
mul.f16x2 r3379, r3376, r3233;
}
{
add.f16x2 r3382, r3373, r3379;
}
{
sub.f16x2 r3385, r1885, r1933;
}
{
mul.f16x2 r3388, r3385, r3231;
}
{
sub.f16x2 r3391, r1901, r1917;
}
{
mul.f16x2 r3394, r3391, r3234;
}
{
add.f16x2 r3397, r3388, r3394;
}
{
add.f16x2 r3400, r3382, r3397;
}
st.local.u32 [rd3+152], r3400;
{
add.f16x2 r3403, r1885, r1933;
}
{
mul.f16x2 r3406, r3403, r3225;
}
{
add.f16x2 r3409, r270, r3406;
}
{
add.f16x2 r3412, r1901, r1917;
}
{
mul.f16x2 r3415, r3412, r3229;
}
{
add.f16x2 r3418, r3409, r3415;
}
{
sub.f16x2 r3421, r1879, r1927;
}
{
mul.f16x2 r3424, r3421, r3227;
}
{
sub.f16x2 r3427, r1895, r1911;
}
{
mul.f16x2 r3430, r3427, r3231;
}
{
add.f16x2 r3433, r3424, r3430;
}
{
add.f16x2 r3436, r3418, r3433;
}
st.local.u32 [rd3+76], r3436;
{
add.f16x2 r3439, r1885, r1933;
}
{
mul.f16x2 r3442, r3439, r3225;
}
{
add.f16x2 r3445, r270, r3442;
}
{
add.f16x2 r3448, r1901, r1917;
}
{
mul.f16x2 r3451, r3448, r3229;
}
{
add.f16x2 r3454, r3445, r3451;
}
{
sub.f16x2 r3457, r1879, r1927;
}
{
mul.f16x2 r3460, r3457, r3227;
}
{
sub.f16x2 r3463, r1895, r1911;
}
{
mul.f16x2 r3466, r3463, r3231;
}
{
add.f16x2 r3469, r3460, r3466;
}
{
sub.f16x2 r3472, r3454, r3469;
}
st.local.u32 [rd3+196], r3472;
{
add.f16x2 r3475, r1885, r1933;
}
{
mul.f16x2 r3478, r3475, r3229;
}
{
add.f16x2 r3481, r270, r3478;
}
{
add.f16x2 r3484, r1901, r1917;
}
{
mul.f16x2 r3487, r3484, r3233;
}
{
add.f16x2 r3490, r3481, r3487;
}
{
sub.f16x2 r3493, r1879, r1927;
}
{
mul.f16x2 r3496, r3493, r3231;
}
{
sub.f16x2 r3499, r1895, r1911;
}
{
mul.f16x2 r3502, r3499, r3234;
}
{
add.f16x2 r3505, r3496, r3502;
}
{
add.f16x2 r3508, r3490, r3505;
}
st.local.u32 [rd3+116], r3508;
{
add.f16x2 r3511, r1885, r1933;
}
{
mul.f16x2 r3514, r3511, r3229;
}
{
add.f16x2 r3517, r270, r3514;
}
{
add.f16x2 r3520, r1901, r1917;
}
{
mul.f16x2 r3523, r3520, r3233;
}
{
add.f16x2 r3526, r3517, r3523;
}
{
sub.f16x2 r3529, r1879, r1927;
}
{
mul.f16x2 r3532, r3529, r3231;
}
{
sub.f16x2 r3535, r1895, r1911;
}
{
mul.f16x2 r3538, r3535, r3234;
}
{
add.f16x2 r3541, r3532, r3538;
}
{
sub.f16x2 r3544, r3526, r3541;
}
st.local.u32 [rd3+156], r3544;
mov.u32 r3552, %tid.x;
mul.wide.u32 rd9, r3552, -776530087;
shr.u64 rd10, rd9, 41;
cvt.u32.u64 r4, rd10;
mul.lo.s32 r3553, r4, 625;
sub.s32 r3, r3552, r3553;
cvt.rn.f32.u32 f221, r3;
mul.f32 f222, f221, 0f39D2D427;
cos.approx.f32 f217, f222;
sin.approx.f32 f223, f222;
neg.f32 f218, f223;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r11310, {low, high};
}
mov.u32 r11309, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11310;
mov.b32 r3574, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11310;
mov.b32 r3576, {high, high};
}
bra.uni LBB1_1;
LBB1_2:
ld.local.u32 r11308, [rd5+60];
LBB1_1:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11310;
mov.b32 r3554, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11310;
mov.b32 r3556, {high, high};
}
mul.wide.u32 rd11, r11309, 8;
add.s64 rd12, rd3, rd11;
add.s64 rd5, rd12, 4;
ld.local.u32 r3559, [rd12+4];
{
mul.f16x2 r3558, r3559, r3556;
}
{
fma.rn.f16x2 r3561, r11308, r3554, r3558;
}
st.local.u32 [rd12], r3561;
{
mul.f16x2 r3565, r11308, r3556;
}
{
neg.f16x2 r3568, r3565;
}
{
fma.rn.f16x2 r3570, r3559, r3554, r3568;
}
st.local.u32 [rd12+4], r3570;
mov.f32 f238, 0fBF800000;
mov.f32 f239, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3578, {low, high};
}
{
mul.f16x2 r3579, r3576, r3578;
}
{
mul.f16x2 r3582, r11310, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11310;
mov.b32 r3585, {high, low};
}
{
fma.rn.f16x2 r3587, r3579, r3585, r3582;
}
ld.local.u32 r3603, [rd12+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3591, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3593, {high, high};
}
ld.local.u32 r3608, [rd12+12];
{
mul.f16x2 r3595, r3608, r3593;
}
{
fma.rn.f16x2 r3598, r3603, r3591, r3595;
}
st.local.u32 [rd12+8], r3598;
{
mul.f16x2 r3602, r3603, r3593;
}
{
neg.f16x2 r3605, r3602;
}
{
fma.rn.f16x2 r3607, r3608, r3591, r3605;
}
st.local.u32 [rd12+12], r3607;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3615, {low, high};
}
{
mul.f16x2 r3616, r3576, r3615;
}
{
mul.f16x2 r3619, r3587, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3587;
mov.b32 r3622, {high, low};
}
{
fma.rn.f16x2 r3624, r3616, r3622, r3619;
}
ld.local.u32 r3640, [rd12+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3628, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3630, {high, high};
}
ld.local.u32 r3645, [rd12+20];
{
mul.f16x2 r3632, r3645, r3630;
}
{
fma.rn.f16x2 r3635, r3640, r3628, r3632;
}
st.local.u32 [rd12+16], r3635;
{
mul.f16x2 r3639, r3640, r3630;
}
{
neg.f16x2 r3642, r3639;
}
{
fma.rn.f16x2 r3644, r3645, r3628, r3642;
}
st.local.u32 [rd12+20], r3644;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3652, {low, high};
}
{
mul.f16x2 r3653, r3576, r3652;
}
{
mul.f16x2 r3656, r3624, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3624;
mov.b32 r3659, {high, low};
}
{
fma.rn.f16x2 r3661, r3653, r3659, r3656;
}
ld.local.u32 r3677, [rd12+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3665, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3667, {high, high};
}
ld.local.u32 r3682, [rd12+28];
{
mul.f16x2 r3669, r3682, r3667;
}
{
fma.rn.f16x2 r3672, r3677, r3665, r3669;
}
st.local.u32 [rd12+24], r3672;
{
mul.f16x2 r3676, r3677, r3667;
}
{
neg.f16x2 r3679, r3676;
}
{
fma.rn.f16x2 r3681, r3682, r3665, r3679;
}
st.local.u32 [rd12+28], r3681;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3689, {low, high};
}
{
mul.f16x2 r3690, r3576, r3689;
}
{
mul.f16x2 r3693, r3661, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3661;
mov.b32 r3696, {high, low};
}
{
fma.rn.f16x2 r3698, r3690, r3696, r3693;
}
ld.local.u32 r3710, [rd12+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3702, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3704, {high, high};
}
ld.local.u32 r3719, [rd12+36];
{
mul.f16x2 r3706, r3719, r3704;
}
{
fma.rn.f16x2 r3709, r3710, r3702, r3706;
}
st.local.u32 [rd12+32], r3709;
{
mul.f16x2 r3713, r3710, r3704;
}
{
neg.f16x2 r3716, r3713;
}
{
fma.rn.f16x2 r3718, r3719, r3702, r3716;
}
st.local.u32 [rd12+36], r3718;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3726, {low, high};
}
{
mul.f16x2 r3727, r3576, r3726;
}
{
mul.f16x2 r3730, r3698, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3698;
mov.b32 r3733, {high, low};
}
{
fma.rn.f16x2 r3735, r3727, r3733, r3730;
}
ld.local.u32 r3747, [rd12+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3739, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3741, {high, high};
}
ld.local.u32 r3744, [rd12+44];
{
mul.f16x2 r3743, r3744, r3741;
}
{
fma.rn.f16x2 r3746, r3747, r3739, r3743;
}
st.local.u32 [rd12+40], r3746;
{
mul.f16x2 r3750, r3747, r3741;
}
{
neg.f16x2 r3753, r3750;
}
{
fma.rn.f16x2 r3755, r3744, r3739, r3753;
}
st.local.u32 [rd12+44], r3755;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3763, {low, high};
}
{
mul.f16x2 r3764, r3576, r3763;
}
{
mul.f16x2 r3767, r3735, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3735;
mov.b32 r3770, {high, low};
}
{
fma.rn.f16x2 r3772, r3764, r3770, r3767;
}
ld.local.u32 r3784, [rd12+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3776, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3778, {high, high};
}
ld.local.u32 r3781, [rd12+52];
{
mul.f16x2 r3780, r3781, r3778;
}
{
fma.rn.f16x2 r3783, r3784, r3776, r3780;
}
st.local.u32 [rd12+48], r3783;
{
mul.f16x2 r3787, r3784, r3778;
}
{
neg.f16x2 r3790, r3787;
}
{
fma.rn.f16x2 r3792, r3781, r3776, r3790;
}
st.local.u32 [rd12+52], r3792;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3800, {low, high};
}
{
mul.f16x2 r3801, r3576, r3800;
}
{
mul.f16x2 r3804, r3772, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3772;
mov.b32 r3807, {high, low};
}
{
fma.rn.f16x2 r3809, r3801, r3807, r3804;
}
ld.local.u32 r3821, [rd12+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3813, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3815, {high, high};
}
ld.local.u32 r3818, [rd12+60];
{
mul.f16x2 r3817, r3818, r3815;
}
{
fma.rn.f16x2 r3820, r3821, r3813, r3817;
}
st.local.u32 [rd12+56], r3820;
{
mul.f16x2 r3824, r3821, r3815;
}
{
neg.f16x2 r3827, r3824;
}
{
fma.rn.f16x2 r3829, r3818, r3813, r3827;
}
st.local.u32 [rd12+60], r3829;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r3837, {low, high};
}
{
mul.f16x2 r3838, r3576, r3837;
}
{
mul.f16x2 r3841, r3809, r3574;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3809;
mov.b32 r3844, {high, low};
}
{
fma.rn.f16x2 r11310, r3838, r3844, r3841;
}
add.s32 r11309, r11309, 8;
setp.eq.s32 p1, r11309, 25;
@p1 bra LBB1_3;
bra.uni LBB1_2;
LBB1_3:
shl.b32 r7378, r1, 2;
mov.u32 r7379, %50;
add.s32 r7380, r7379, r7378;
mad.lo.s32 r12, r4, 62500, r7380;
barrier.sync 0;
mad.lo.s32 r7381, r3, 100, r12;
ld.local.u32 r7382, [rd3];
st.shared.u32 [r7381], r7382;
ld.local.u32 r7383, [rd4+4];
st.shared.u32 [r7381+4], r7383;
ld.local.u32 r7384, [rd4+12];
st.shared.u32 [r7381+8], r7384;
ld.local.u32 r7385, [rd4+20];
st.shared.u32 [r7381+12], r7385;
ld.local.u32 r7386, [rd4+28];
st.shared.u32 [r7381+16], r7386;
ld.local.u32 r7387, [rd4+36];
st.shared.u32 [r7381+20], r7387;
ld.local.u32 r7388, [rd4+44];
st.shared.u32 [r7381+24], r7388;
ld.local.u32 r7389, [rd4+52];
st.shared.u32 [r7381+28], r7389;
ld.local.u32 r7390, [rd4+60];
st.shared.u32 [r7381+32], r7390;
ld.local.u32 r7391, [rd4+68];
st.shared.u32 [r7381+36], r7391;
ld.local.u32 r7392, [rd4+76];
st.shared.u32 [r7381+40], r7392;
ld.local.u32 r7393, [rd4+84];
st.shared.u32 [r7381+44], r7393;
ld.local.u32 r7394, [rd4+92];
st.shared.u32 [r7381+48], r7394;
ld.local.u32 r7395, [rd4+100];
st.shared.u32 [r7381+52], r7395;
ld.local.u32 r7396, [rd4+108];
st.shared.u32 [r7381+56], r7396;
ld.local.u32 r7397, [rd4+116];
st.shared.u32 [r7381+60], r7397;
ld.local.u32 r7398, [rd4+124];
st.shared.u32 [r7381+64], r7398;
ld.local.u32 r7399, [rd4+132];
st.shared.u32 [r7381+68], r7399;
ld.local.u32 r7400, [rd4+140];
st.shared.u32 [r7381+72], r7400;
ld.local.u32 r7401, [rd4+148];
st.shared.u32 [r7381+76], r7401;
ld.local.u32 r7402, [rd4+156];
st.shared.u32 [r7381+80], r7402;
ld.local.u32 r7403, [rd4+164];
st.shared.u32 [r7381+84], r7403;
ld.local.u32 r7404, [rd4+172];
st.shared.u32 [r7381+88], r7404;
ld.local.u32 r7405, [rd4+180];
st.shared.u32 [r7381+92], r7405;
ld.local.u32 r7406, [rd4+188];
st.shared.u32 [r7381+96], r7406;
barrier.sync 0;
mad.lo.s32 r13, r3, -96, r7381;
ld.shared.u32 r3864, [r13];
ld.shared.u32 r4186, [r13+2500];
ld.shared.u32 r4508, [r13+5000];
ld.shared.u32 r4830, [r13+7500];
ld.shared.u32 r5152, [r13+10000];
ld.shared.u32 r3861, [r13+12500];
ld.shared.u32 r4183, [r13+15000];
ld.shared.u32 r4505, [r13+17500];
ld.shared.u32 r4827, [r13+20000];
ld.shared.u32 r5149, [r13+22500];
ld.shared.u32 r3867, [r13+25000];
ld.shared.u32 r4189, [r13+27500];
ld.shared.u32 r4511, [r13+30000];
ld.shared.u32 r4833, [r13+32500];
ld.shared.u32 r5155, [r13+35000];
ld.shared.u32 r3868, [r13+37500];
ld.shared.u32 r4190, [r13+40000];
ld.shared.u32 r4512, [r13+42500];
ld.shared.u32 r4834, [r13+45000];
ld.shared.u32 r5156, [r13+47500];
ld.shared.u32 r3862, [r13+50000];
ld.shared.u32 r4184, [r13+52500];
ld.shared.u32 r4506, [r13+55000];
ld.shared.u32 r4828, [r13+57500];
ld.shared.u32 r5150, [r13+60000];
barrier.sync 0;
ld.local.u32 r7407, [rd4];
st.shared.u32 [r7381], r7407;
ld.local.u32 r7408, [rd4+8];
st.shared.u32 [r7381+4], r7408;
ld.local.u32 r7409, [rd4+16];
st.shared.u32 [r7381+8], r7409;
ld.local.u32 r7410, [rd4+24];
st.shared.u32 [r7381+12], r7410;
ld.local.u32 r7411, [rd4+32];
st.shared.u32 [r7381+16], r7411;
ld.local.u32 r7412, [rd4+40];
st.shared.u32 [r7381+20], r7412;
ld.local.u32 r7413, [rd4+48];
st.shared.u32 [r7381+24], r7413;
ld.local.u32 r7414, [rd4+56];
st.shared.u32 [r7381+28], r7414;
ld.local.u32 r7415, [rd4+64];
st.shared.u32 [r7381+32], r7415;
ld.local.u32 r7416, [rd4+72];
st.shared.u32 [r7381+36], r7416;
ld.local.u32 r7417, [rd4+80];
st.shared.u32 [r7381+40], r7417;
ld.local.u32 r7418, [rd4+88];
st.shared.u32 [r7381+44], r7418;
ld.local.u32 r7419, [rd4+96];
st.shared.u32 [r7381+48], r7419;
ld.local.u32 r7420, [rd4+104];
st.shared.u32 [r7381+52], r7420;
ld.local.u32 r7421, [rd4+112];
st.shared.u32 [r7381+56], r7421;
ld.local.u32 r7422, [rd4+120];
st.shared.u32 [r7381+60], r7422;
ld.local.u32 r7423, [rd4+128];
st.shared.u32 [r7381+64], r7423;
ld.local.u32 r7424, [rd4+136];
st.shared.u32 [r7381+68], r7424;
ld.local.u32 r7425, [rd4+144];
st.shared.u32 [r7381+72], r7425;
ld.local.u32 r7426, [rd4+152];
st.shared.u32 [r7381+76], r7426;
ld.local.u32 r7427, [rd4+160];
st.shared.u32 [r7381+80], r7427;
ld.local.u32 r7428, [rd4+168];
st.shared.u32 [r7381+84], r7428;
ld.local.u32 r7429, [rd4+176];
st.shared.u32 [r7381+88], r7429;
ld.local.u32 r7430, [rd4+184];
st.shared.u32 [r7381+92], r7430;
ld.local.u32 r7431, [rd4+192];
st.shared.u32 [r7381+96], r7431;
barrier.sync 0;
ld.shared.u32 r3876, [r13];
ld.shared.u32 r4198, [r13+2500];
ld.shared.u32 r4520, [r13+5000];
ld.shared.u32 r4842, [r13+7500];
ld.shared.u32 r5164, [r13+10000];
ld.shared.u32 r3873, [r13+12500];
ld.shared.u32 r4195, [r13+15000];
ld.shared.u32 r4517, [r13+17500];
ld.shared.u32 r4839, [r13+20000];
ld.shared.u32 r5161, [r13+22500];
ld.shared.u32 r3879, [r13+25000];
ld.shared.u32 r4201, [r13+27500];
ld.shared.u32 r4523, [r13+30000];
ld.shared.u32 r4845, [r13+32500];
ld.shared.u32 r5167, [r13+35000];
ld.shared.u32 r3880, [r13+37500];
ld.shared.u32 r4202, [r13+40000];
ld.shared.u32 r4524, [r13+42500];
ld.shared.u32 r4846, [r13+45000];
ld.shared.u32 r5168, [r13+47500];
ld.shared.u32 r3874, [r13+50000];
ld.shared.u32 r4196, [r13+52500];
ld.shared.u32 r4518, [r13+55000];
ld.shared.u32 r4840, [r13+57500];
ld.shared.u32 r5162, [r13+60000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3850, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3851, {low, high};
}
{
neg.f16x2 r3852, r3851;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r3854, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r3855, {low, high};
}
{
neg.f16x2 r3856, r3855;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r3858, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r3859, {low, high};
}
{
add.f16x2 r3860, r3861, r3862;
}
{
add.f16x2 r3863, r3864, r3860;
}
{
add.f16x2 r3866, r3867, r3868;
}
{
add.f16x2 r3869, r3863, r3866;
}
{
add.f16x2 r3872, r3873, r3874;
}
{
add.f16x2 r3875, r3876, r3872;
}
{
add.f16x2 r3878, r3879, r3880;
}
{
add.f16x2 r3881, r3875, r3878;
}
{
add.f16x2 r3884, r3861, r3862;
}
{
mul.f16x2 r3887, r3884, r3850;
}
{
add.f16x2 r3890, r3864, r3887;
}
{
add.f16x2 r3893, r3867, r3868;
}
{
mul.f16x2 r3896, r3893, r3854;
}
{
add.f16x2 r3899, r3890, r3896;
}
{
sub.f16x2 r3902, r3873, r3874;
}
{
mul.f16x2 r3905, r3902, r3852;
}
{
sub.f16x2 r3908, r3879, r3880;
}
{
mul.f16x2 r3911, r3908, r3856;
}
{
add.f16x2 r3914, r3905, r3911;
}
{
sub.f16x2 r3917, r3899, r3914;
}
{
add.f16x2 r3920, r3861, r3862;
}
{
mul.f16x2 r3923, r3920, r3850;
}
{
add.f16x2 r3926, r3864, r3923;
}
{
add.f16x2 r3929, r3867, r3868;
}
{
mul.f16x2 r3932, r3929, r3854;
}
{
add.f16x2 r3935, r3926, r3932;
}
{
sub.f16x2 r3938, r3873, r3874;
}
{
mul.f16x2 r3941, r3938, r3852;
}
{
sub.f16x2 r3944, r3879, r3880;
}
{
mul.f16x2 r3947, r3944, r3856;
}
{
add.f16x2 r3950, r3941, r3947;
}
{
add.f16x2 r3953, r3935, r3950;
}
{
add.f16x2 r3956, r3861, r3862;
}
{
mul.f16x2 r3959, r3956, r3854;
}
{
add.f16x2 r3962, r3864, r3959;
}
{
add.f16x2 r3965, r3867, r3868;
}
{
mul.f16x2 r3968, r3965, r3858;
}
{
add.f16x2 r3971, r3962, r3968;
}
{
sub.f16x2 r3974, r3873, r3874;
}
{
mul.f16x2 r3977, r3974, r3856;
}
{
sub.f16x2 r3980, r3879, r3880;
}
{
mul.f16x2 r3983, r3980, r3859;
}
{
add.f16x2 r3986, r3977, r3983;
}
{
sub.f16x2 r3989, r3971, r3986;
}
{
add.f16x2 r3992, r3861, r3862;
}
{
mul.f16x2 r3995, r3992, r3854;
}
{
add.f16x2 r3998, r3864, r3995;
}
{
add.f16x2 r4001, r3867, r3868;
}
{
mul.f16x2 r4004, r4001, r3858;
}
{
add.f16x2 r4007, r3998, r4004;
}
{
sub.f16x2 r4010, r3873, r3874;
}
{
mul.f16x2 r4013, r4010, r3856;
}
{
sub.f16x2 r4016, r3879, r3880;
}
{
mul.f16x2 r4019, r4016, r3859;
}
{
add.f16x2 r4022, r4013, r4019;
}
{
add.f16x2 r4025, r4007, r4022;
}
{
add.f16x2 r4028, r3873, r3874;
}
{
mul.f16x2 r4031, r4028, r3850;
}
{
add.f16x2 r4034, r3876, r4031;
}
{
add.f16x2 r4037, r3879, r3880;
}
{
mul.f16x2 r4040, r4037, r3854;
}
{
add.f16x2 r4043, r4034, r4040;
}
{
sub.f16x2 r4046, r3861, r3862;
}
{
mul.f16x2 r4049, r4046, r3852;
}
{
sub.f16x2 r4052, r3867, r3868;
}
{
mul.f16x2 r4055, r4052, r3856;
}
{
add.f16x2 r4058, r4049, r4055;
}
{
add.f16x2 r4061, r4043, r4058;
}
{
add.f16x2 r4064, r3873, r3874;
}
{
mul.f16x2 r4067, r4064, r3850;
}
{
add.f16x2 r4070, r3876, r4067;
}
{
add.f16x2 r4073, r3879, r3880;
}
{
mul.f16x2 r4076, r4073, r3854;
}
{
add.f16x2 r4079, r4070, r4076;
}
{
sub.f16x2 r4082, r3861, r3862;
}
{
mul.f16x2 r4085, r4082, r3852;
}
{
sub.f16x2 r4088, r3867, r3868;
}
{
mul.f16x2 r4091, r4088, r3856;
}
{
add.f16x2 r4094, r4085, r4091;
}
{
sub.f16x2 r4097, r4079, r4094;
}
{
add.f16x2 r4100, r3873, r3874;
}
{
mul.f16x2 r4103, r4100, r3854;
}
{
add.f16x2 r4106, r3876, r4103;
}
{
add.f16x2 r4109, r3879, r3880;
}
{
mul.f16x2 r4112, r4109, r3858;
}
{
add.f16x2 r4115, r4106, r4112;
}
{
sub.f16x2 r4118, r3861, r3862;
}
{
mul.f16x2 r4121, r4118, r3856;
}
{
sub.f16x2 r4124, r3867, r3868;
}
{
mul.f16x2 r4127, r4124, r3859;
}
{
add.f16x2 r4130, r4121, r4127;
}
{
add.f16x2 r4133, r4115, r4130;
}
{
add.f16x2 r4136, r3873, r3874;
}
{
mul.f16x2 r4139, r4136, r3854;
}
{
add.f16x2 r4142, r3876, r4139;
}
{
add.f16x2 r4145, r3879, r3880;
}
{
mul.f16x2 r4148, r4145, r3858;
}
{
add.f16x2 r4151, r4142, r4148;
}
{
sub.f16x2 r4154, r3861, r3862;
}
{
mul.f16x2 r4157, r4154, r3856;
}
{
sub.f16x2 r4160, r3867, r3868;
}
{
mul.f16x2 r4163, r4160, r3859;
}
{
add.f16x2 r4166, r4157, r4163;
}
{
sub.f16x2 r4169, r4151, r4166;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4172, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4173, {low, high};
}
{
neg.f16x2 r4174, r4173;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4176, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4177, {low, high};
}
{
neg.f16x2 r4178, r4177;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4180, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4181, {low, high};
}
{
add.f16x2 r4182, r4183, r4184;
}
{
add.f16x2 r4185, r4186, r4182;
}
{
add.f16x2 r4188, r4189, r4190;
}
{
add.f16x2 r4191, r4185, r4188;
}
{
add.f16x2 r4194, r4195, r4196;
}
{
add.f16x2 r4197, r4198, r4194;
}
{
add.f16x2 r4200, r4201, r4202;
}
{
add.f16x2 r4203, r4197, r4200;
}
{
add.f16x2 r4206, r4183, r4184;
}
{
mul.f16x2 r4209, r4206, r4172;
}
{
add.f16x2 r4212, r4186, r4209;
}
{
add.f16x2 r4215, r4189, r4190;
}
{
mul.f16x2 r4218, r4215, r4176;
}
{
add.f16x2 r4221, r4212, r4218;
}
{
sub.f16x2 r4224, r4195, r4196;
}
{
mul.f16x2 r4227, r4224, r4174;
}
{
sub.f16x2 r4230, r4201, r4202;
}
{
mul.f16x2 r4233, r4230, r4178;
}
{
add.f16x2 r4236, r4227, r4233;
}
{
sub.f16x2 r4239, r4221, r4236;
}
{
add.f16x2 r4242, r4183, r4184;
}
{
mul.f16x2 r4245, r4242, r4172;
}
{
add.f16x2 r4248, r4186, r4245;
}
{
add.f16x2 r4251, r4189, r4190;
}
{
mul.f16x2 r4254, r4251, r4176;
}
{
add.f16x2 r4257, r4248, r4254;
}
{
sub.f16x2 r4260, r4195, r4196;
}
{
mul.f16x2 r4263, r4260, r4174;
}
{
sub.f16x2 r4266, r4201, r4202;
}
{
mul.f16x2 r4269, r4266, r4178;
}
{
add.f16x2 r4272, r4263, r4269;
}
{
add.f16x2 r4275, r4257, r4272;
}
{
add.f16x2 r4278, r4183, r4184;
}
{
mul.f16x2 r4281, r4278, r4176;
}
{
add.f16x2 r4284, r4186, r4281;
}
{
add.f16x2 r4287, r4189, r4190;
}
{
mul.f16x2 r4290, r4287, r4180;
}
{
add.f16x2 r4293, r4284, r4290;
}
{
sub.f16x2 r4296, r4195, r4196;
}
{
mul.f16x2 r4299, r4296, r4178;
}
{
sub.f16x2 r4302, r4201, r4202;
}
{
mul.f16x2 r4305, r4302, r4181;
}
{
add.f16x2 r4308, r4299, r4305;
}
{
sub.f16x2 r4311, r4293, r4308;
}
{
add.f16x2 r4314, r4183, r4184;
}
{
mul.f16x2 r4317, r4314, r4176;
}
{
add.f16x2 r4320, r4186, r4317;
}
{
add.f16x2 r4323, r4189, r4190;
}
{
mul.f16x2 r4326, r4323, r4180;
}
{
add.f16x2 r4329, r4320, r4326;
}
{
sub.f16x2 r4332, r4195, r4196;
}
{
mul.f16x2 r4335, r4332, r4178;
}
{
sub.f16x2 r4338, r4201, r4202;
}
{
mul.f16x2 r4341, r4338, r4181;
}
{
add.f16x2 r4344, r4335, r4341;
}
{
add.f16x2 r4347, r4329, r4344;
}
{
add.f16x2 r4350, r4195, r4196;
}
{
mul.f16x2 r4353, r4350, r4172;
}
{
add.f16x2 r4356, r4198, r4353;
}
{
add.f16x2 r4359, r4201, r4202;
}
{
mul.f16x2 r4362, r4359, r4176;
}
{
add.f16x2 r4365, r4356, r4362;
}
{
sub.f16x2 r4368, r4183, r4184;
}
{
mul.f16x2 r4371, r4368, r4174;
}
{
sub.f16x2 r4374, r4189, r4190;
}
{
mul.f16x2 r4377, r4374, r4178;
}
{
add.f16x2 r4380, r4371, r4377;
}
{
add.f16x2 r4383, r4365, r4380;
}
{
add.f16x2 r4386, r4195, r4196;
}
{
mul.f16x2 r4389, r4386, r4172;
}
{
add.f16x2 r4392, r4198, r4389;
}
{
add.f16x2 r4395, r4201, r4202;
}
{
mul.f16x2 r4398, r4395, r4176;
}
{
add.f16x2 r4401, r4392, r4398;
}
{
sub.f16x2 r4404, r4183, r4184;
}
{
mul.f16x2 r4407, r4404, r4174;
}
{
sub.f16x2 r4410, r4189, r4190;
}
{
mul.f16x2 r4413, r4410, r4178;
}
{
add.f16x2 r4416, r4407, r4413;
}
{
sub.f16x2 r4419, r4401, r4416;
}
{
add.f16x2 r4422, r4195, r4196;
}
{
mul.f16x2 r4425, r4422, r4176;
}
{
add.f16x2 r4428, r4198, r4425;
}
{
add.f16x2 r4431, r4201, r4202;
}
{
mul.f16x2 r4434, r4431, r4180;
}
{
add.f16x2 r4437, r4428, r4434;
}
{
sub.f16x2 r4440, r4183, r4184;
}
{
mul.f16x2 r4443, r4440, r4178;
}
{
sub.f16x2 r4446, r4189, r4190;
}
{
mul.f16x2 r4449, r4446, r4181;
}
{
add.f16x2 r4452, r4443, r4449;
}
{
add.f16x2 r4455, r4437, r4452;
}
{
add.f16x2 r4458, r4195, r4196;
}
{
mul.f16x2 r4461, r4458, r4176;
}
{
add.f16x2 r4464, r4198, r4461;
}
{
add.f16x2 r4467, r4201, r4202;
}
{
mul.f16x2 r4470, r4467, r4180;
}
{
add.f16x2 r4473, r4464, r4470;
}
{
sub.f16x2 r4476, r4183, r4184;
}
{
mul.f16x2 r4479, r4476, r4178;
}
{
sub.f16x2 r4482, r4189, r4190;
}
{
mul.f16x2 r4485, r4482, r4181;
}
{
add.f16x2 r4488, r4479, r4485;
}
{
sub.f16x2 r4491, r4473, r4488;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4494, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4495, {low, high};
}
{
neg.f16x2 r4496, r4495;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4498, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4499, {low, high};
}
{
neg.f16x2 r4500, r4499;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4502, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4503, {low, high};
}
{
add.f16x2 r4504, r4505, r4506;
}
{
add.f16x2 r4507, r4508, r4504;
}
{
add.f16x2 r4510, r4511, r4512;
}
{
add.f16x2 r4513, r4507, r4510;
}
{
add.f16x2 r4516, r4517, r4518;
}
{
add.f16x2 r4519, r4520, r4516;
}
{
add.f16x2 r4522, r4523, r4524;
}
{
add.f16x2 r4525, r4519, r4522;
}
{
add.f16x2 r4528, r4505, r4506;
}
{
mul.f16x2 r4531, r4528, r4494;
}
{
add.f16x2 r4534, r4508, r4531;
}
{
add.f16x2 r4537, r4511, r4512;
}
{
mul.f16x2 r4540, r4537, r4498;
}
{
add.f16x2 r4543, r4534, r4540;
}
{
sub.f16x2 r4546, r4517, r4518;
}
{
mul.f16x2 r4549, r4546, r4496;
}
{
sub.f16x2 r4552, r4523, r4524;
}
{
mul.f16x2 r4555, r4552, r4500;
}
{
add.f16x2 r4558, r4549, r4555;
}
{
sub.f16x2 r4561, r4543, r4558;
}
{
add.f16x2 r4564, r4505, r4506;
}
{
mul.f16x2 r4567, r4564, r4494;
}
{
add.f16x2 r4570, r4508, r4567;
}
{
add.f16x2 r4573, r4511, r4512;
}
{
mul.f16x2 r4576, r4573, r4498;
}
{
add.f16x2 r4579, r4570, r4576;
}
{
sub.f16x2 r4582, r4517, r4518;
}
{
mul.f16x2 r4585, r4582, r4496;
}
{
sub.f16x2 r4588, r4523, r4524;
}
{
mul.f16x2 r4591, r4588, r4500;
}
{
add.f16x2 r4594, r4585, r4591;
}
{
add.f16x2 r4597, r4579, r4594;
}
{
add.f16x2 r4600, r4505, r4506;
}
{
mul.f16x2 r4603, r4600, r4498;
}
{
add.f16x2 r4606, r4508, r4603;
}
{
add.f16x2 r4609, r4511, r4512;
}
{
mul.f16x2 r4612, r4609, r4502;
}
{
add.f16x2 r4615, r4606, r4612;
}
{
sub.f16x2 r4618, r4517, r4518;
}
{
mul.f16x2 r4621, r4618, r4500;
}
{
sub.f16x2 r4624, r4523, r4524;
}
{
mul.f16x2 r4627, r4624, r4503;
}
{
add.f16x2 r4630, r4621, r4627;
}
{
sub.f16x2 r4633, r4615, r4630;
}
{
add.f16x2 r4636, r4505, r4506;
}
{
mul.f16x2 r4639, r4636, r4498;
}
{
add.f16x2 r4642, r4508, r4639;
}
{
add.f16x2 r4645, r4511, r4512;
}
{
mul.f16x2 r4648, r4645, r4502;
}
{
add.f16x2 r4651, r4642, r4648;
}
{
sub.f16x2 r4654, r4517, r4518;
}
{
mul.f16x2 r4657, r4654, r4500;
}
{
sub.f16x2 r4660, r4523, r4524;
}
{
mul.f16x2 r4663, r4660, r4503;
}
{
add.f16x2 r4666, r4657, r4663;
}
{
add.f16x2 r4669, r4651, r4666;
}
{
add.f16x2 r4672, r4517, r4518;
}
{
mul.f16x2 r4675, r4672, r4494;
}
{
add.f16x2 r4678, r4520, r4675;
}
{
add.f16x2 r4681, r4523, r4524;
}
{
mul.f16x2 r4684, r4681, r4498;
}
{
add.f16x2 r4687, r4678, r4684;
}
{
sub.f16x2 r4690, r4505, r4506;
}
{
mul.f16x2 r4693, r4690, r4496;
}
{
sub.f16x2 r4696, r4511, r4512;
}
{
mul.f16x2 r4699, r4696, r4500;
}
{
add.f16x2 r4702, r4693, r4699;
}
{
add.f16x2 r4705, r4687, r4702;
}
{
add.f16x2 r4708, r4517, r4518;
}
{
mul.f16x2 r4711, r4708, r4494;
}
{
add.f16x2 r4714, r4520, r4711;
}
{
add.f16x2 r4717, r4523, r4524;
}
{
mul.f16x2 r4720, r4717, r4498;
}
{
add.f16x2 r4723, r4714, r4720;
}
{
sub.f16x2 r4726, r4505, r4506;
}
{
mul.f16x2 r4729, r4726, r4496;
}
{
sub.f16x2 r4732, r4511, r4512;
}
{
mul.f16x2 r4735, r4732, r4500;
}
{
add.f16x2 r4738, r4729, r4735;
}
{
sub.f16x2 r4741, r4723, r4738;
}
{
add.f16x2 r4744, r4517, r4518;
}
{
mul.f16x2 r4747, r4744, r4498;
}
{
add.f16x2 r4750, r4520, r4747;
}
{
add.f16x2 r4753, r4523, r4524;
}
{
mul.f16x2 r4756, r4753, r4502;
}
{
add.f16x2 r4759, r4750, r4756;
}
{
sub.f16x2 r4762, r4505, r4506;
}
{
mul.f16x2 r4765, r4762, r4500;
}
{
sub.f16x2 r4768, r4511, r4512;
}
{
mul.f16x2 r4771, r4768, r4503;
}
{
add.f16x2 r4774, r4765, r4771;
}
{
add.f16x2 r4777, r4759, r4774;
}
{
add.f16x2 r4780, r4517, r4518;
}
{
mul.f16x2 r4783, r4780, r4498;
}
{
add.f16x2 r4786, r4520, r4783;
}
{
add.f16x2 r4789, r4523, r4524;
}
{
mul.f16x2 r4792, r4789, r4502;
}
{
add.f16x2 r4795, r4786, r4792;
}
{
sub.f16x2 r4798, r4505, r4506;
}
{
mul.f16x2 r4801, r4798, r4500;
}
{
sub.f16x2 r4804, r4511, r4512;
}
{
mul.f16x2 r4807, r4804, r4503;
}
{
add.f16x2 r4810, r4801, r4807;
}
{
sub.f16x2 r4813, r4795, r4810;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4816, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4817, {low, high};
}
{
neg.f16x2 r4818, r4817;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r4820, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r4821, {low, high};
}
{
neg.f16x2 r4822, r4821;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r4824, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r4825, {low, high};
}
{
add.f16x2 r4826, r4827, r4828;
}
{
add.f16x2 r4829, r4830, r4826;
}
{
add.f16x2 r4832, r4833, r4834;
}
{
add.f16x2 r4835, r4829, r4832;
}
{
add.f16x2 r4838, r4839, r4840;
}
{
add.f16x2 r4841, r4842, r4838;
}
{
add.f16x2 r4844, r4845, r4846;
}
{
add.f16x2 r4847, r4841, r4844;
}
{
add.f16x2 r4850, r4827, r4828;
}
{
mul.f16x2 r4853, r4850, r4816;
}
{
add.f16x2 r4856, r4830, r4853;
}
{
add.f16x2 r4859, r4833, r4834;
}
{
mul.f16x2 r4862, r4859, r4820;
}
{
add.f16x2 r4865, r4856, r4862;
}
{
sub.f16x2 r4868, r4839, r4840;
}
{
mul.f16x2 r4871, r4868, r4818;
}
{
sub.f16x2 r4874, r4845, r4846;
}
{
mul.f16x2 r4877, r4874, r4822;
}
{
add.f16x2 r4880, r4871, r4877;
}
{
sub.f16x2 r4883, r4865, r4880;
}
{
add.f16x2 r4886, r4827, r4828;
}
{
mul.f16x2 r4889, r4886, r4816;
}
{
add.f16x2 r4892, r4830, r4889;
}
{
add.f16x2 r4895, r4833, r4834;
}
{
mul.f16x2 r4898, r4895, r4820;
}
{
add.f16x2 r4901, r4892, r4898;
}
{
sub.f16x2 r4904, r4839, r4840;
}
{
mul.f16x2 r4907, r4904, r4818;
}
{
sub.f16x2 r4910, r4845, r4846;
}
{
mul.f16x2 r4913, r4910, r4822;
}
{
add.f16x2 r4916, r4907, r4913;
}
{
add.f16x2 r4919, r4901, r4916;
}
{
add.f16x2 r4922, r4827, r4828;
}
{
mul.f16x2 r4925, r4922, r4820;
}
{
add.f16x2 r4928, r4830, r4925;
}
{
add.f16x2 r4931, r4833, r4834;
}
{
mul.f16x2 r4934, r4931, r4824;
}
{
add.f16x2 r4937, r4928, r4934;
}
{
sub.f16x2 r4940, r4839, r4840;
}
{
mul.f16x2 r4943, r4940, r4822;
}
{
sub.f16x2 r4946, r4845, r4846;
}
{
mul.f16x2 r4949, r4946, r4825;
}
{
add.f16x2 r4952, r4943, r4949;
}
{
sub.f16x2 r4955, r4937, r4952;
}
{
add.f16x2 r4958, r4827, r4828;
}
{
mul.f16x2 r4961, r4958, r4820;
}
{
add.f16x2 r4964, r4830, r4961;
}
{
add.f16x2 r4967, r4833, r4834;
}
{
mul.f16x2 r4970, r4967, r4824;
}
{
add.f16x2 r4973, r4964, r4970;
}
{
sub.f16x2 r4976, r4839, r4840;
}
{
mul.f16x2 r4979, r4976, r4822;
}
{
sub.f16x2 r4982, r4845, r4846;
}
{
mul.f16x2 r4985, r4982, r4825;
}
{
add.f16x2 r4988, r4979, r4985;
}
{
add.f16x2 r4991, r4973, r4988;
}
{
add.f16x2 r4994, r4839, r4840;
}
{
mul.f16x2 r4997, r4994, r4816;
}
{
add.f16x2 r5000, r4842, r4997;
}
{
add.f16x2 r5003, r4845, r4846;
}
{
mul.f16x2 r5006, r5003, r4820;
}
{
add.f16x2 r5009, r5000, r5006;
}
{
sub.f16x2 r5012, r4827, r4828;
}
{
mul.f16x2 r5015, r5012, r4818;
}
{
sub.f16x2 r5018, r4833, r4834;
}
{
mul.f16x2 r5021, r5018, r4822;
}
{
add.f16x2 r5024, r5015, r5021;
}
{
add.f16x2 r5027, r5009, r5024;
}
{
add.f16x2 r5030, r4839, r4840;
}
{
mul.f16x2 r5033, r5030, r4816;
}
{
add.f16x2 r5036, r4842, r5033;
}
{
add.f16x2 r5039, r4845, r4846;
}
{
mul.f16x2 r5042, r5039, r4820;
}
{
add.f16x2 r5045, r5036, r5042;
}
{
sub.f16x2 r5048, r4827, r4828;
}
{
mul.f16x2 r5051, r5048, r4818;
}
{
sub.f16x2 r5054, r4833, r4834;
}
{
mul.f16x2 r5057, r5054, r4822;
}
{
add.f16x2 r5060, r5051, r5057;
}
{
sub.f16x2 r5063, r5045, r5060;
}
{
add.f16x2 r5066, r4839, r4840;
}
{
mul.f16x2 r5069, r5066, r4820;
}
{
add.f16x2 r5072, r4842, r5069;
}
{
add.f16x2 r5075, r4845, r4846;
}
{
mul.f16x2 r5078, r5075, r4824;
}
{
add.f16x2 r5081, r5072, r5078;
}
{
sub.f16x2 r5084, r4827, r4828;
}
{
mul.f16x2 r5087, r5084, r4822;
}
{
sub.f16x2 r5090, r4833, r4834;
}
{
mul.f16x2 r5093, r5090, r4825;
}
{
add.f16x2 r5096, r5087, r5093;
}
{
add.f16x2 r5099, r5081, r5096;
}
{
add.f16x2 r5102, r4839, r4840;
}
{
mul.f16x2 r5105, r5102, r4820;
}
{
add.f16x2 r5108, r4842, r5105;
}
{
add.f16x2 r5111, r4845, r4846;
}
{
mul.f16x2 r5114, r5111, r4824;
}
{
add.f16x2 r5117, r5108, r5114;
}
{
sub.f16x2 r5120, r4827, r4828;
}
{
mul.f16x2 r5123, r5120, r4822;
}
{
sub.f16x2 r5126, r4833, r4834;
}
{
mul.f16x2 r5129, r5126, r4825;
}
{
add.f16x2 r5132, r5123, r5129;
}
{
sub.f16x2 r5135, r5117, r5132;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5138, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5139, {low, high};
}
{
neg.f16x2 r5140, r5139;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5142, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5143, {low, high};
}
{
neg.f16x2 r5144, r5143;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5146, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5147, {low, high};
}
{
add.f16x2 r5148, r5149, r5150;
}
{
add.f16x2 r5151, r5152, r5148;
}
{
add.f16x2 r5154, r5155, r5156;
}
{
add.f16x2 r5157, r5151, r5154;
}
{
add.f16x2 r5160, r5161, r5162;
}
{
add.f16x2 r5163, r5164, r5160;
}
{
add.f16x2 r5166, r5167, r5168;
}
{
add.f16x2 r5169, r5163, r5166;
}
{
add.f16x2 r5172, r5149, r5150;
}
{
mul.f16x2 r5175, r5172, r5138;
}
{
add.f16x2 r5178, r5152, r5175;
}
{
add.f16x2 r5181, r5155, r5156;
}
{
mul.f16x2 r5184, r5181, r5142;
}
{
add.f16x2 r5187, r5178, r5184;
}
{
sub.f16x2 r5190, r5161, r5162;
}
{
mul.f16x2 r5193, r5190, r5140;
}
{
sub.f16x2 r5196, r5167, r5168;
}
{
mul.f16x2 r5199, r5196, r5144;
}
{
add.f16x2 r5202, r5193, r5199;
}
{
sub.f16x2 r5205, r5187, r5202;
}
{
add.f16x2 r5208, r5149, r5150;
}
{
mul.f16x2 r5211, r5208, r5138;
}
{
add.f16x2 r5214, r5152, r5211;
}
{
add.f16x2 r5217, r5155, r5156;
}
{
mul.f16x2 r5220, r5217, r5142;
}
{
add.f16x2 r5223, r5214, r5220;
}
{
sub.f16x2 r5226, r5161, r5162;
}
{
mul.f16x2 r5229, r5226, r5140;
}
{
sub.f16x2 r5232, r5167, r5168;
}
{
mul.f16x2 r5235, r5232, r5144;
}
{
add.f16x2 r5238, r5229, r5235;
}
{
add.f16x2 r5241, r5223, r5238;
}
{
add.f16x2 r5244, r5149, r5150;
}
{
mul.f16x2 r5247, r5244, r5142;
}
{
add.f16x2 r5250, r5152, r5247;
}
{
add.f16x2 r5253, r5155, r5156;
}
{
mul.f16x2 r5256, r5253, r5146;
}
{
add.f16x2 r5259, r5250, r5256;
}
{
sub.f16x2 r5262, r5161, r5162;
}
{
mul.f16x2 r5265, r5262, r5144;
}
{
sub.f16x2 r5268, r5167, r5168;
}
{
mul.f16x2 r5271, r5268, r5147;
}
{
add.f16x2 r5274, r5265, r5271;
}
{
sub.f16x2 r5277, r5259, r5274;
}
{
add.f16x2 r5280, r5149, r5150;
}
{
mul.f16x2 r5283, r5280, r5142;
}
{
add.f16x2 r5286, r5152, r5283;
}
{
add.f16x2 r5289, r5155, r5156;
}
{
mul.f16x2 r5292, r5289, r5146;
}
{
add.f16x2 r5295, r5286, r5292;
}
{
sub.f16x2 r5298, r5161, r5162;
}
{
mul.f16x2 r5301, r5298, r5144;
}
{
sub.f16x2 r5304, r5167, r5168;
}
{
mul.f16x2 r5307, r5304, r5147;
}
{
add.f16x2 r5310, r5301, r5307;
}
{
add.f16x2 r5313, r5295, r5310;
}
{
add.f16x2 r5316, r5161, r5162;
}
{
mul.f16x2 r5319, r5316, r5138;
}
{
add.f16x2 r5322, r5164, r5319;
}
{
add.f16x2 r5325, r5167, r5168;
}
{
mul.f16x2 r5328, r5325, r5142;
}
{
add.f16x2 r5331, r5322, r5328;
}
{
sub.f16x2 r5334, r5149, r5150;
}
{
mul.f16x2 r5337, r5334, r5140;
}
{
sub.f16x2 r5340, r5155, r5156;
}
{
mul.f16x2 r5343, r5340, r5144;
}
{
add.f16x2 r5346, r5337, r5343;
}
{
add.f16x2 r5349, r5331, r5346;
}
{
add.f16x2 r5352, r5161, r5162;
}
{
mul.f16x2 r5355, r5352, r5138;
}
{
add.f16x2 r5358, r5164, r5355;
}
{
add.f16x2 r5361, r5167, r5168;
}
{
mul.f16x2 r5364, r5361, r5142;
}
{
add.f16x2 r5367, r5358, r5364;
}
{
sub.f16x2 r5370, r5149, r5150;
}
{
mul.f16x2 r5373, r5370, r5140;
}
{
sub.f16x2 r5376, r5155, r5156;
}
{
mul.f16x2 r5379, r5376, r5144;
}
{
add.f16x2 r5382, r5373, r5379;
}
{
sub.f16x2 r5385, r5367, r5382;
}
{
add.f16x2 r5388, r5161, r5162;
}
{
mul.f16x2 r5391, r5388, r5142;
}
{
add.f16x2 r5394, r5164, r5391;
}
{
add.f16x2 r5397, r5167, r5168;
}
{
mul.f16x2 r5400, r5397, r5146;
}
{
add.f16x2 r5403, r5394, r5400;
}
{
sub.f16x2 r5406, r5149, r5150;
}
{
mul.f16x2 r5409, r5406, r5144;
}
{
sub.f16x2 r5412, r5155, r5156;
}
{
mul.f16x2 r5415, r5412, r5147;
}
{
add.f16x2 r5418, r5409, r5415;
}
{
add.f16x2 r5421, r5403, r5418;
}
{
add.f16x2 r5424, r5161, r5162;
}
{
mul.f16x2 r5427, r5424, r5142;
}
{
add.f16x2 r5430, r5164, r5427;
}
{
add.f16x2 r5433, r5167, r5168;
}
{
mul.f16x2 r5436, r5433, r5146;
}
{
add.f16x2 r5439, r5430, r5436;
}
{
sub.f16x2 r5442, r5149, r5150;
}
{
mul.f16x2 r5445, r5442, r5144;
}
{
sub.f16x2 r5448, r5155, r5156;
}
{
mul.f16x2 r5451, r5448, r5147;
}
{
add.f16x2 r5454, r5445, r5451;
}
{
sub.f16x2 r5457, r5439, r5454;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r5460, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r5461, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r5462, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r5463, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r5464, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r5465, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r5466, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r5467, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5470, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r5471, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r5474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r5475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r5477, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5482, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r5483, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r5490, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r5491, {low, high};
}
{
mul.f16x2 r5508, r4239, r5460;
}
{
mul.f16x2 r5511, r4383, r5461;
}
{
sub.f16x2 r5514, r5508, r5511;
}
{
mul.f16x2 r5517, r4239, r5461;
}
{
fma.rn.f16x2 r5520, r4383, r5460, r5517;
}
{
mul.f16x2 r5524, r4561, r5462;
}
{
mul.f16x2 r5527, r4705, r5463;
}
{
sub.f16x2 r5530, r5524, r5527;
}
{
mul.f16x2 r5533, r4561, r5463;
}
{
fma.rn.f16x2 r5536, r4705, r5462, r5533;
}
{
mul.f16x2 r5540, r4883, r5464;
}
{
mul.f16x2 r5543, r5027, r5465;
}
{
sub.f16x2 r5546, r5540, r5543;
}
{
mul.f16x2 r5549, r4883, r5465;
}
{
fma.rn.f16x2 r5552, r5027, r5464, r5549;
}
{
mul.f16x2 r5556, r5205, r5466;
}
{
mul.f16x2 r5559, r5349, r5467;
}
{
sub.f16x2 r5562, r5556, r5559;
}
{
mul.f16x2 r5565, r5205, r5467;
}
{
fma.rn.f16x2 r5568, r5349, r5466, r5565;
}
{
mul.f16x2 r5572, r4311, r5462;
}
{
mul.f16x2 r5575, r4455, r5463;
}
{
sub.f16x2 r5578, r5572, r5575;
}
{
mul.f16x2 r5581, r4311, r5463;
}
{
fma.rn.f16x2 r5584, r4455, r5462, r5581;
}
{
mul.f16x2 r5588, r4633, r5466;
}
{
mul.f16x2 r5591, r4777, r5467;
}
{
sub.f16x2 r5594, r5588, r5591;
}
{
mul.f16x2 r5597, r4633, r5467;
}
{
fma.rn.f16x2 r5600, r4777, r5466, r5597;
}
{
mul.f16x2 r5604, r4955, r5470;
}
{
mul.f16x2 r5607, r5099, r5471;
}
{
sub.f16x2 r5610, r5604, r5607;
}
{
mul.f16x2 r5613, r4955, r5471;
}
{
fma.rn.f16x2 r5616, r5099, r5470, r5613;
}
{
mul.f16x2 r5620, r5277, r5474;
}
{
mul.f16x2 r5623, r5421, r5475;
}
{
sub.f16x2 r5626, r5620, r5623;
}
{
mul.f16x2 r5629, r5277, r5475;
}
{
fma.rn.f16x2 r5632, r5421, r5474, r5629;
}
{
mul.f16x2 r5636, r4347, r5464;
}
{
mul.f16x2 r5639, r4491, r5465;
}
{
sub.f16x2 r5642, r5636, r5639;
}
{
mul.f16x2 r5645, r4347, r5465;
}
{
fma.rn.f16x2 r5648, r4491, r5464, r5645;
}
{
mul.f16x2 r5652, r4669, r5470;
}
{
mul.f16x2 r5655, r4813, r5471;
}
{
sub.f16x2 r5658, r5652, r5655;
}
{
mul.f16x2 r5661, r4669, r5471;
}
{
fma.rn.f16x2 r5664, r4813, r5470, r5661;
}
{
mul.f16x2 r5668, r4991, r5476;
}
{
mul.f16x2 r5671, r5135, r5477;
}
{
sub.f16x2 r5674, r5668, r5671;
}
{
mul.f16x2 r5677, r4991, r5477;
}
{
fma.rn.f16x2 r5680, r5135, r5476, r5677;
}
{
mul.f16x2 r5684, r5313, r5482;
}
{
mul.f16x2 r5687, r5457, r5483;
}
{
sub.f16x2 r5690, r5684, r5687;
}
{
mul.f16x2 r5693, r5313, r5483;
}
{
fma.rn.f16x2 r5696, r5457, r5482, r5693;
}
{
mul.f16x2 r5700, r4275, r5466;
}
{
mul.f16x2 r5703, r4419, r5467;
}
{
sub.f16x2 r5706, r5700, r5703;
}
{
mul.f16x2 r5709, r4275, r5467;
}
{
fma.rn.f16x2 r5712, r4419, r5466, r5709;
}
{
mul.f16x2 r5716, r4597, r5474;
}
{
mul.f16x2 r5719, r4741, r5475;
}
{
sub.f16x2 r5722, r5716, r5719;
}
{
mul.f16x2 r5725, r4597, r5475;
}
{
fma.rn.f16x2 r5728, r4741, r5474, r5725;
}
{
mul.f16x2 r5732, r4919, r5482;
}
{
mul.f16x2 r5735, r5063, r5483;
}
{
sub.f16x2 r5738, r5732, r5735;
}
{
mul.f16x2 r5741, r4919, r5483;
}
{
fma.rn.f16x2 r5744, r5063, r5482, r5741;
}
{
mul.f16x2 r5748, r5241, r5490;
}
{
mul.f16x2 r5751, r5385, r5491;
}
{
sub.f16x2 r5754, r5748, r5751;
}
{
mul.f16x2 r5757, r5241, r5491;
}
{
fma.rn.f16x2 r5760, r5385, r5490, r5757;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5764, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5765, {low, high};
}
{
neg.f16x2 r5766, r5765;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r5768, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r5769, {low, high};
}
{
neg.f16x2 r5770, r5769;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r5772, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r5773, {low, high};
}
{
add.f16x2 r5774, r4191, r5157;
}
{
add.f16x2 r5777, r3869, r5774;
}
{
add.f16x2 r5780, r4513, r4835;
}
{
add.f16x2 r5783, r5777, r5780;
}
st.local.u32 [rd3], r5783;
{
add.f16x2 r5786, r4203, r5169;
}
{
add.f16x2 r5789, r3881, r5786;
}
{
add.f16x2 r5792, r4525, r4847;
}
{
add.f16x2 r5795, r5789, r5792;
}
st.local.u32 [rd4], r5795;
{
add.f16x2 r5798, r4191, r5157;
}
{
mul.f16x2 r5801, r5798, r5764;
}
{
add.f16x2 r5804, r3869, r5801;
}
{
add.f16x2 r5807, r4513, r4835;
}
{
mul.f16x2 r5810, r5807, r5768;
}
{
add.f16x2 r5813, r5804, r5810;
}
{
sub.f16x2 r5816, r4203, r5169;
}
{
mul.f16x2 r5819, r5816, r5766;
}
{
sub.f16x2 r5822, r4525, r4847;
}
{
mul.f16x2 r5825, r5822, r5770;
}
{
add.f16x2 r5828, r5819, r5825;
}
{
sub.f16x2 r5831, r5813, r5828;
}
st.local.u32 [rd4+36], r5831;
{
add.f16x2 r5834, r4191, r5157;
}
{
mul.f16x2 r5837, r5834, r5764;
}
{
add.f16x2 r5840, r3869, r5837;
}
{
add.f16x2 r5843, r4513, r4835;
}
{
mul.f16x2 r5846, r5843, r5768;
}
{
add.f16x2 r5849, r5840, r5846;
}
{
sub.f16x2 r5852, r4203, r5169;
}
{
mul.f16x2 r5855, r5852, r5766;
}
{
sub.f16x2 r5858, r4525, r4847;
}
{
mul.f16x2 r5861, r5858, r5770;
}
{
add.f16x2 r5864, r5855, r5861;
}
{
add.f16x2 r5867, r5849, r5864;
}
st.local.u32 [rd4+156], r5867;
{
add.f16x2 r5870, r4191, r5157;
}
{
mul.f16x2 r5873, r5870, r5768;
}
{
add.f16x2 r5876, r3869, r5873;
}
{
add.f16x2 r5879, r4513, r4835;
}
{
mul.f16x2 r5882, r5879, r5772;
}
{
add.f16x2 r5885, r5876, r5882;
}
{
sub.f16x2 r5888, r4203, r5169;
}
{
mul.f16x2 r5891, r5888, r5770;
}
{
sub.f16x2 r5894, r4525, r4847;
}
{
mul.f16x2 r5897, r5894, r5773;
}
{
add.f16x2 r5900, r5891, r5897;
}
{
sub.f16x2 r5903, r5885, r5900;
}
st.local.u32 [rd4+76], r5903;
{
add.f16x2 r5906, r4191, r5157;
}
{
mul.f16x2 r5909, r5906, r5768;
}
{
add.f16x2 r5912, r3869, r5909;
}
{
add.f16x2 r5915, r4513, r4835;
}
{
mul.f16x2 r5918, r5915, r5772;
}
{
add.f16x2 r5921, r5912, r5918;
}
{
sub.f16x2 r5924, r4203, r5169;
}
{
mul.f16x2 r5927, r5924, r5770;
}
{
sub.f16x2 r5930, r4525, r4847;
}
{
mul.f16x2 r5933, r5930, r5773;
}
{
add.f16x2 r5936, r5927, r5933;
}
{
add.f16x2 r5939, r5921, r5936;
}
st.local.u32 [rd4+116], r5939;
{
add.f16x2 r5942, r4203, r5169;
}
{
mul.f16x2 r5945, r5942, r5764;
}
{
add.f16x2 r5948, r3881, r5945;
}
{
add.f16x2 r5951, r4525, r4847;
}
{
mul.f16x2 r5954, r5951, r5768;
}
{
add.f16x2 r5957, r5948, r5954;
}
{
sub.f16x2 r5960, r4191, r5157;
}
{
mul.f16x2 r5963, r5960, r5766;
}
{
sub.f16x2 r5966, r4513, r4835;
}
{
mul.f16x2 r5969, r5966, r5770;
}
{
add.f16x2 r5972, r5963, r5969;
}
{
add.f16x2 r5975, r5957, r5972;
}
st.local.u32 [rd4+40], r5975;
{
add.f16x2 r5978, r4203, r5169;
}
{
mul.f16x2 r5981, r5978, r5764;
}
{
add.f16x2 r5984, r3881, r5981;
}
{
add.f16x2 r5987, r4525, r4847;
}
{
mul.f16x2 r5990, r5987, r5768;
}
{
add.f16x2 r5993, r5984, r5990;
}
{
sub.f16x2 r5996, r4191, r5157;
}
{
mul.f16x2 r5999, r5996, r5766;
}
{
sub.f16x2 r6002, r4513, r4835;
}
{
mul.f16x2 r6005, r6002, r5770;
}
{
add.f16x2 r6008, r5999, r6005;
}
{
sub.f16x2 r6011, r5993, r6008;
}
st.local.u32 [rd4+160], r6011;
{
add.f16x2 r6014, r4203, r5169;
}
{
mul.f16x2 r6017, r6014, r5768;
}
{
add.f16x2 r6020, r3881, r6017;
}
{
add.f16x2 r6023, r4525, r4847;
}
{
mul.f16x2 r6026, r6023, r5772;
}
{
add.f16x2 r6029, r6020, r6026;
}
{
sub.f16x2 r6032, r4191, r5157;
}
{
mul.f16x2 r6035, r6032, r5770;
}
{
sub.f16x2 r6038, r4513, r4835;
}
{
mul.f16x2 r6041, r6038, r5773;
}
{
add.f16x2 r6044, r6035, r6041;
}
{
add.f16x2 r6047, r6029, r6044;
}
st.local.u32 [rd4+80], r6047;
{
add.f16x2 r6050, r4203, r5169;
}
{
mul.f16x2 r6053, r6050, r5768;
}
{
add.f16x2 r6056, r3881, r6053;
}
{
add.f16x2 r6059, r4525, r4847;
}
{
mul.f16x2 r6062, r6059, r5772;
}
{
add.f16x2 r6065, r6056, r6062;
}
{
sub.f16x2 r6068, r4191, r5157;
}
{
mul.f16x2 r6071, r6068, r5770;
}
{
sub.f16x2 r6074, r4513, r4835;
}
{
mul.f16x2 r6077, r6074, r5773;
}
{
add.f16x2 r6080, r6071, r6077;
}
{
sub.f16x2 r6083, r6065, r6080;
}
st.local.u32 [rd4+120], r6083;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6086, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6087, {low, high};
}
{
neg.f16x2 r6088, r6087;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6090, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6091, {low, high};
}
{
neg.f16x2 r6092, r6091;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6094, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6095, {low, high};
}
{
add.f16x2 r6096, r5514, r5562;
}
{
add.f16x2 r6099, r3917, r6096;
}
{
add.f16x2 r6102, r5530, r5546;
}
{
add.f16x2 r11311, r6099, r6102;
}
st.local.u32 [rd4+4], r11311;
{
add.f16x2 r6108, r5520, r5568;
}
{
add.f16x2 r6111, r4061, r6108;
}
{
add.f16x2 r6114, r5536, r5552;
}
{
add.f16x2 r6117, r6111, r6114;
}
st.local.u32 [rd4+8], r6117;
{
add.f16x2 r6120, r5514, r5562;
}
{
mul.f16x2 r6123, r6120, r6086;
}
{
add.f16x2 r6126, r3917, r6123;
}
{
add.f16x2 r6129, r5530, r5546;
}
{
mul.f16x2 r6132, r6129, r6090;
}
{
add.f16x2 r6135, r6126, r6132;
}
{
sub.f16x2 r6138, r5520, r5568;
}
{
mul.f16x2 r6141, r6138, r6088;
}
{
sub.f16x2 r6144, r5536, r5552;
}
{
mul.f16x2 r6147, r6144, r6092;
}
{
add.f16x2 r6150, r6141, r6147;
}
{
sub.f16x2 r6153, r6135, r6150;
}
st.local.u32 [rd4+44], r6153;
{
add.f16x2 r6156, r5514, r5562;
}
{
mul.f16x2 r6159, r6156, r6086;
}
{
add.f16x2 r6162, r3917, r6159;
}
{
add.f16x2 r6165, r5530, r5546;
}
{
mul.f16x2 r6168, r6165, r6090;
}
{
add.f16x2 r6171, r6162, r6168;
}
{
sub.f16x2 r6174, r5520, r5568;
}
{
mul.f16x2 r6177, r6174, r6088;
}
{
sub.f16x2 r6180, r5536, r5552;
}
{
mul.f16x2 r6183, r6180, r6092;
}
{
add.f16x2 r6186, r6177, r6183;
}
{
add.f16x2 r6189, r6171, r6186;
}
st.local.u32 [rd4+164], r6189;
{
add.f16x2 r6192, r5514, r5562;
}
{
mul.f16x2 r6195, r6192, r6090;
}
{
add.f16x2 r6198, r3917, r6195;
}
{
add.f16x2 r6201, r5530, r5546;
}
{
mul.f16x2 r6204, r6201, r6094;
}
{
add.f16x2 r6207, r6198, r6204;
}
{
sub.f16x2 r6210, r5520, r5568;
}
{
mul.f16x2 r6213, r6210, r6092;
}
{
sub.f16x2 r6216, r5536, r5552;
}
{
mul.f16x2 r6219, r6216, r6095;
}
{
add.f16x2 r6222, r6213, r6219;
}
{
sub.f16x2 r6225, r6207, r6222;
}
st.local.u32 [rd4+84], r6225;
{
add.f16x2 r6228, r5514, r5562;
}
{
mul.f16x2 r6231, r6228, r6090;
}
{
add.f16x2 r6234, r3917, r6231;
}
{
add.f16x2 r6237, r5530, r5546;
}
{
mul.f16x2 r6240, r6237, r6094;
}
{
add.f16x2 r6243, r6234, r6240;
}
{
sub.f16x2 r6246, r5520, r5568;
}
{
mul.f16x2 r6249, r6246, r6092;
}
{
sub.f16x2 r6252, r5536, r5552;
}
{
mul.f16x2 r6255, r6252, r6095;
}
{
add.f16x2 r6258, r6249, r6255;
}
{
add.f16x2 r6261, r6243, r6258;
}
st.local.u32 [rd4+124], r6261;
{
add.f16x2 r6264, r5520, r5568;
}
{
mul.f16x2 r6267, r6264, r6086;
}
{
add.f16x2 r6270, r4061, r6267;
}
{
add.f16x2 r6273, r5536, r5552;
}
{
mul.f16x2 r6276, r6273, r6090;
}
{
add.f16x2 r6279, r6270, r6276;
}
{
sub.f16x2 r6282, r5514, r5562;
}
{
mul.f16x2 r6285, r6282, r6088;
}
{
sub.f16x2 r6288, r5530, r5546;
}
{
mul.f16x2 r6291, r6288, r6092;
}
{
add.f16x2 r6294, r6285, r6291;
}
{
add.f16x2 r6297, r6279, r6294;
}
st.local.u32 [rd4+48], r6297;
{
add.f16x2 r6300, r5520, r5568;
}
{
mul.f16x2 r6303, r6300, r6086;
}
{
add.f16x2 r6306, r4061, r6303;
}
{
add.f16x2 r6309, r5536, r5552;
}
{
mul.f16x2 r6312, r6309, r6090;
}
{
add.f16x2 r6315, r6306, r6312;
}
{
sub.f16x2 r6318, r5514, r5562;
}
{
mul.f16x2 r6321, r6318, r6088;
}
{
sub.f16x2 r6324, r5530, r5546;
}
{
mul.f16x2 r6327, r6324, r6092;
}
{
add.f16x2 r6330, r6321, r6327;
}
{
sub.f16x2 r6333, r6315, r6330;
}
st.local.u32 [rd4+168], r6333;
{
add.f16x2 r6336, r5520, r5568;
}
{
mul.f16x2 r6339, r6336, r6090;
}
{
add.f16x2 r6342, r4061, r6339;
}
{
add.f16x2 r6345, r5536, r5552;
}
{
mul.f16x2 r6348, r6345, r6094;
}
{
add.f16x2 r6351, r6342, r6348;
}
{
sub.f16x2 r6354, r5514, r5562;
}
{
mul.f16x2 r6357, r6354, r6092;
}
{
sub.f16x2 r6360, r5530, r5546;
}
{
mul.f16x2 r6363, r6360, r6095;
}
{
add.f16x2 r6366, r6357, r6363;
}
{
add.f16x2 r6369, r6351, r6366;
}
st.local.u32 [rd4+88], r6369;
{
add.f16x2 r6372, r5520, r5568;
}
{
mul.f16x2 r6375, r6372, r6090;
}
{
add.f16x2 r6378, r4061, r6375;
}
{
add.f16x2 r6381, r5536, r5552;
}
{
mul.f16x2 r6384, r6381, r6094;
}
{
add.f16x2 r6387, r6378, r6384;
}
{
sub.f16x2 r6390, r5514, r5562;
}
{
mul.f16x2 r6393, r6390, r6092;
}
{
sub.f16x2 r6396, r5530, r5546;
}
{
mul.f16x2 r6399, r6396, r6095;
}
{
add.f16x2 r6402, r6393, r6399;
}
{
sub.f16x2 r6405, r6387, r6402;
}
st.local.u32 [rd4+128], r6405;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6408, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6409, {low, high};
}
{
neg.f16x2 r6410, r6409;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6412, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6413, {low, high};
}
{
neg.f16x2 r6414, r6413;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6417, {low, high};
}
{
add.f16x2 r6418, r5578, r5626;
}
{
add.f16x2 r6421, r3989, r6418;
}
{
add.f16x2 r6424, r5594, r5610;
}
{
add.f16x2 r6427, r6421, r6424;
}
st.local.u32 [rd4+12], r6427;
{
add.f16x2 r6430, r5584, r5632;
}
{
add.f16x2 r6433, r4133, r6430;
}
{
add.f16x2 r6436, r5600, r5616;
}
{
add.f16x2 r6439, r6433, r6436;
}
st.local.u32 [rd4+16], r6439;
{
add.f16x2 r6442, r5578, r5626;
}
{
mul.f16x2 r6445, r6442, r6408;
}
{
add.f16x2 r6448, r3989, r6445;
}
{
add.f16x2 r6451, r5594, r5610;
}
{
mul.f16x2 r6454, r6451, r6412;
}
{
add.f16x2 r6457, r6448, r6454;
}
{
sub.f16x2 r6460, r5584, r5632;
}
{
mul.f16x2 r6463, r6460, r6410;
}
{
sub.f16x2 r6466, r5600, r5616;
}
{
mul.f16x2 r6469, r6466, r6414;
}
{
add.f16x2 r6472, r6463, r6469;
}
{
sub.f16x2 r6475, r6457, r6472;
}
st.local.u32 [rd4+52], r6475;
{
add.f16x2 r6478, r5578, r5626;
}
{
mul.f16x2 r6481, r6478, r6408;
}
{
add.f16x2 r6484, r3989, r6481;
}
{
add.f16x2 r6487, r5594, r5610;
}
{
mul.f16x2 r6490, r6487, r6412;
}
{
add.f16x2 r6493, r6484, r6490;
}
{
sub.f16x2 r6496, r5584, r5632;
}
{
mul.f16x2 r6499, r6496, r6410;
}
{
sub.f16x2 r6502, r5600, r5616;
}
{
mul.f16x2 r6505, r6502, r6414;
}
{
add.f16x2 r6508, r6499, r6505;
}
{
add.f16x2 r6511, r6493, r6508;
}
st.local.u32 [rd4+172], r6511;
{
add.f16x2 r6514, r5578, r5626;
}
{
mul.f16x2 r6517, r6514, r6412;
}
{
add.f16x2 r6520, r3989, r6517;
}
{
add.f16x2 r6523, r5594, r5610;
}
{
mul.f16x2 r6526, r6523, r6416;
}
{
add.f16x2 r6529, r6520, r6526;
}
{
sub.f16x2 r6532, r5584, r5632;
}
{
mul.f16x2 r6535, r6532, r6414;
}
{
sub.f16x2 r6538, r5600, r5616;
}
{
mul.f16x2 r6541, r6538, r6417;
}
{
add.f16x2 r6544, r6535, r6541;
}
{
sub.f16x2 r6547, r6529, r6544;
}
st.local.u32 [rd4+92], r6547;
{
add.f16x2 r6550, r5578, r5626;
}
{
mul.f16x2 r6553, r6550, r6412;
}
{
add.f16x2 r6556, r3989, r6553;
}
{
add.f16x2 r6559, r5594, r5610;
}
{
mul.f16x2 r6562, r6559, r6416;
}
{
add.f16x2 r6565, r6556, r6562;
}
{
sub.f16x2 r6568, r5584, r5632;
}
{
mul.f16x2 r6571, r6568, r6414;
}
{
sub.f16x2 r6574, r5600, r5616;
}
{
mul.f16x2 r6577, r6574, r6417;
}
{
add.f16x2 r6580, r6571, r6577;
}
{
add.f16x2 r6583, r6565, r6580;
}
st.local.u32 [rd4+132], r6583;
{
add.f16x2 r6586, r5584, r5632;
}
{
mul.f16x2 r6589, r6586, r6408;
}
{
add.f16x2 r6592, r4133, r6589;
}
{
add.f16x2 r6595, r5600, r5616;
}
{
mul.f16x2 r6598, r6595, r6412;
}
{
add.f16x2 r6601, r6592, r6598;
}
{
sub.f16x2 r6604, r5578, r5626;
}
{
mul.f16x2 r6607, r6604, r6410;
}
{
sub.f16x2 r6610, r5594, r5610;
}
{
mul.f16x2 r6613, r6610, r6414;
}
{
add.f16x2 r6616, r6607, r6613;
}
{
add.f16x2 r6619, r6601, r6616;
}
st.local.u32 [rd4+56], r6619;
{
add.f16x2 r6622, r5584, r5632;
}
{
mul.f16x2 r6625, r6622, r6408;
}
{
add.f16x2 r6628, r4133, r6625;
}
{
add.f16x2 r6631, r5600, r5616;
}
{
mul.f16x2 r6634, r6631, r6412;
}
{
add.f16x2 r6637, r6628, r6634;
}
{
sub.f16x2 r6640, r5578, r5626;
}
{
mul.f16x2 r6643, r6640, r6410;
}
{
sub.f16x2 r6646, r5594, r5610;
}
{
mul.f16x2 r6649, r6646, r6414;
}
{
add.f16x2 r6652, r6643, r6649;
}
{
sub.f16x2 r6655, r6637, r6652;
}
st.local.u32 [rd4+176], r6655;
{
add.f16x2 r6658, r5584, r5632;
}
{
mul.f16x2 r6661, r6658, r6412;
}
{
add.f16x2 r6664, r4133, r6661;
}
{
add.f16x2 r6667, r5600, r5616;
}
{
mul.f16x2 r6670, r6667, r6416;
}
{
add.f16x2 r6673, r6664, r6670;
}
{
sub.f16x2 r6676, r5578, r5626;
}
{
mul.f16x2 r6679, r6676, r6414;
}
{
sub.f16x2 r6682, r5594, r5610;
}
{
mul.f16x2 r6685, r6682, r6417;
}
{
add.f16x2 r6688, r6679, r6685;
}
{
add.f16x2 r6691, r6673, r6688;
}
st.local.u32 [rd4+96], r6691;
{
add.f16x2 r6694, r5584, r5632;
}
{
mul.f16x2 r6697, r6694, r6412;
}
{
add.f16x2 r6700, r4133, r6697;
}
{
add.f16x2 r6703, r5600, r5616;
}
{
mul.f16x2 r6706, r6703, r6416;
}
{
add.f16x2 r6709, r6700, r6706;
}
{
sub.f16x2 r6712, r5578, r5626;
}
{
mul.f16x2 r6715, r6712, r6414;
}
{
sub.f16x2 r6718, r5594, r5610;
}
{
mul.f16x2 r6721, r6718, r6417;
}
{
add.f16x2 r6724, r6715, r6721;
}
{
sub.f16x2 r6727, r6709, r6724;
}
st.local.u32 [rd4+136], r6727;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6730, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6731, {low, high};
}
{
neg.f16x2 r6732, r6731;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r6734, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r6735, {low, high};
}
{
neg.f16x2 r6736, r6735;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r6738, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r6739, {low, high};
}
{
add.f16x2 r6740, r5642, r5690;
}
{
add.f16x2 r6743, r4025, r6740;
}
{
add.f16x2 r6746, r5658, r5674;
}
{
add.f16x2 r6749, r6743, r6746;
}
st.local.u32 [rd4+20], r6749;
{
add.f16x2 r6752, r5648, r5696;
}
{
add.f16x2 r6755, r4169, r6752;
}
{
add.f16x2 r6758, r5664, r5680;
}
{
add.f16x2 r6761, r6755, r6758;
}
st.local.u32 [rd4+24], r6761;
{
add.f16x2 r6764, r5642, r5690;
}
{
mul.f16x2 r6767, r6764, r6730;
}
{
add.f16x2 r6770, r4025, r6767;
}
{
add.f16x2 r6773, r5658, r5674;
}
{
mul.f16x2 r6776, r6773, r6734;
}
{
add.f16x2 r6779, r6770, r6776;
}
{
sub.f16x2 r6782, r5648, r5696;
}
{
mul.f16x2 r6785, r6782, r6732;
}
{
sub.f16x2 r6788, r5664, r5680;
}
{
mul.f16x2 r6791, r6788, r6736;
}
{
add.f16x2 r6794, r6785, r6791;
}
{
sub.f16x2 r6797, r6779, r6794;
}
st.local.u32 [rd4+60], r6797;
{
add.f16x2 r6800, r5642, r5690;
}
{
mul.f16x2 r6803, r6800, r6730;
}
{
add.f16x2 r6806, r4025, r6803;
}
{
add.f16x2 r6809, r5658, r5674;
}
{
mul.f16x2 r6812, r6809, r6734;
}
{
add.f16x2 r6815, r6806, r6812;
}
{
sub.f16x2 r6818, r5648, r5696;
}
{
mul.f16x2 r6821, r6818, r6732;
}
{
sub.f16x2 r6824, r5664, r5680;
}
{
mul.f16x2 r6827, r6824, r6736;
}
{
add.f16x2 r6830, r6821, r6827;
}
{
add.f16x2 r6833, r6815, r6830;
}
st.local.u32 [rd4+180], r6833;
{
add.f16x2 r6836, r5642, r5690;
}
{
mul.f16x2 r6839, r6836, r6734;
}
{
add.f16x2 r6842, r4025, r6839;
}
{
add.f16x2 r6845, r5658, r5674;
}
{
mul.f16x2 r6848, r6845, r6738;
}
{
add.f16x2 r6851, r6842, r6848;
}
{
sub.f16x2 r6854, r5648, r5696;
}
{
mul.f16x2 r6857, r6854, r6736;
}
{
sub.f16x2 r6860, r5664, r5680;
}
{
mul.f16x2 r6863, r6860, r6739;
}
{
add.f16x2 r6866, r6857, r6863;
}
{
sub.f16x2 r6869, r6851, r6866;
}
st.local.u32 [rd4+100], r6869;
{
add.f16x2 r6872, r5642, r5690;
}
{
mul.f16x2 r6875, r6872, r6734;
}
{
add.f16x2 r6878, r4025, r6875;
}
{
add.f16x2 r6881, r5658, r5674;
}
{
mul.f16x2 r6884, r6881, r6738;
}
{
add.f16x2 r6887, r6878, r6884;
}
{
sub.f16x2 r6890, r5648, r5696;
}
{
mul.f16x2 r6893, r6890, r6736;
}
{
sub.f16x2 r6896, r5664, r5680;
}
{
mul.f16x2 r6899, r6896, r6739;
}
{
add.f16x2 r6902, r6893, r6899;
}
{
add.f16x2 r6905, r6887, r6902;
}
st.local.u32 [rd4+140], r6905;
{
add.f16x2 r6908, r5648, r5696;
}
{
mul.f16x2 r6911, r6908, r6730;
}
{
add.f16x2 r6914, r4169, r6911;
}
{
add.f16x2 r6917, r5664, r5680;
}
{
mul.f16x2 r6920, r6917, r6734;
}
{
add.f16x2 r6923, r6914, r6920;
}
{
sub.f16x2 r6926, r5642, r5690;
}
{
mul.f16x2 r6929, r6926, r6732;
}
{
sub.f16x2 r6932, r5658, r5674;
}
{
mul.f16x2 r6935, r6932, r6736;
}
{
add.f16x2 r6938, r6929, r6935;
}
{
add.f16x2 r6941, r6923, r6938;
}
st.local.u32 [rd4+64], r6941;
{
add.f16x2 r6944, r5648, r5696;
}
{
mul.f16x2 r6947, r6944, r6730;
}
{
add.f16x2 r6950, r4169, r6947;
}
{
add.f16x2 r6953, r5664, r5680;
}
{
mul.f16x2 r6956, r6953, r6734;
}
{
add.f16x2 r6959, r6950, r6956;
}
{
sub.f16x2 r6962, r5642, r5690;
}
{
mul.f16x2 r6965, r6962, r6732;
}
{
sub.f16x2 r6968, r5658, r5674;
}
{
mul.f16x2 r6971, r6968, r6736;
}
{
add.f16x2 r6974, r6965, r6971;
}
{
sub.f16x2 r6977, r6959, r6974;
}
st.local.u32 [rd4+184], r6977;
{
add.f16x2 r6980, r5648, r5696;
}
{
mul.f16x2 r6983, r6980, r6734;
}
{
add.f16x2 r6986, r4169, r6983;
}
{
add.f16x2 r6989, r5664, r5680;
}
{
mul.f16x2 r6992, r6989, r6738;
}
{
add.f16x2 r6995, r6986, r6992;
}
{
sub.f16x2 r6998, r5642, r5690;
}
{
mul.f16x2 r7001, r6998, r6736;
}
{
sub.f16x2 r7004, r5658, r5674;
}
{
mul.f16x2 r7007, r7004, r6739;
}
{
add.f16x2 r7010, r7001, r7007;
}
{
add.f16x2 r7013, r6995, r7010;
}
st.local.u32 [rd4+104], r7013;
{
add.f16x2 r7016, r5648, r5696;
}
{
mul.f16x2 r7019, r7016, r6734;
}
{
add.f16x2 r7022, r4169, r7019;
}
{
add.f16x2 r7025, r5664, r5680;
}
{
mul.f16x2 r7028, r7025, r6738;
}
{
add.f16x2 r7031, r7022, r7028;
}
{
sub.f16x2 r7034, r5642, r5690;
}
{
mul.f16x2 r7037, r7034, r6736;
}
{
sub.f16x2 r7040, r5658, r5674;
}
{
mul.f16x2 r7043, r7040, r6739;
}
{
add.f16x2 r7046, r7037, r7043;
}
{
sub.f16x2 r7049, r7031, r7046;
}
st.local.u32 [rd4+144], r7049;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7052, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7053, {low, high};
}
{
neg.f16x2 r7054, r7053;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7056, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7057, {low, high};
}
{
neg.f16x2 r7058, r7057;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7060, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7061, {low, high};
}
{
add.f16x2 r7062, r5706, r5754;
}
{
add.f16x2 r7065, r3953, r7062;
}
{
add.f16x2 r7068, r5722, r5738;
}
{
add.f16x2 r7071, r7065, r7068;
}
st.local.u32 [rd4+28], r7071;
{
add.f16x2 r7074, r5712, r5760;
}
{
add.f16x2 r7077, r4097, r7074;
}
{
add.f16x2 r7080, r5728, r5744;
}
{
add.f16x2 r7083, r7077, r7080;
}
st.local.u32 [rd4+32], r7083;
{
add.f16x2 r7086, r5706, r5754;
}
{
mul.f16x2 r7089, r7086, r7052;
}
{
add.f16x2 r7092, r3953, r7089;
}
{
add.f16x2 r7095, r5722, r5738;
}
{
mul.f16x2 r7098, r7095, r7056;
}
{
add.f16x2 r7101, r7092, r7098;
}
{
sub.f16x2 r7104, r5712, r5760;
}
{
mul.f16x2 r7107, r7104, r7054;
}
{
sub.f16x2 r7110, r5728, r5744;
}
{
mul.f16x2 r7113, r7110, r7058;
}
{
add.f16x2 r7116, r7107, r7113;
}
{
sub.f16x2 r7119, r7101, r7116;
}
st.local.u32 [rd4+68], r7119;
{
add.f16x2 r7122, r5706, r5754;
}
{
mul.f16x2 r7125, r7122, r7052;
}
{
add.f16x2 r7128, r3953, r7125;
}
{
add.f16x2 r7131, r5722, r5738;
}
{
mul.f16x2 r7134, r7131, r7056;
}
{
add.f16x2 r7137, r7128, r7134;
}
{
sub.f16x2 r7140, r5712, r5760;
}
{
mul.f16x2 r7143, r7140, r7054;
}
{
sub.f16x2 r7146, r5728, r5744;
}
{
mul.f16x2 r7149, r7146, r7058;
}
{
add.f16x2 r7152, r7143, r7149;
}
{
add.f16x2 r7155, r7137, r7152;
}
st.local.u32 [rd4+188], r7155;
{
add.f16x2 r7158, r5706, r5754;
}
{
mul.f16x2 r7161, r7158, r7056;
}
{
add.f16x2 r7164, r3953, r7161;
}
{
add.f16x2 r7167, r5722, r5738;
}
{
mul.f16x2 r7170, r7167, r7060;
}
{
add.f16x2 r7173, r7164, r7170;
}
{
sub.f16x2 r7176, r5712, r5760;
}
{
mul.f16x2 r7179, r7176, r7058;
}
{
sub.f16x2 r7182, r5728, r5744;
}
{
mul.f16x2 r7185, r7182, r7061;
}
{
add.f16x2 r7188, r7179, r7185;
}
{
sub.f16x2 r7191, r7173, r7188;
}
st.local.u32 [rd4+108], r7191;
{
add.f16x2 r7194, r5706, r5754;
}
{
mul.f16x2 r7197, r7194, r7056;
}
{
add.f16x2 r7200, r3953, r7197;
}
{
add.f16x2 r7203, r5722, r5738;
}
{
mul.f16x2 r7206, r7203, r7060;
}
{
add.f16x2 r7209, r7200, r7206;
}
{
sub.f16x2 r7212, r5712, r5760;
}
{
mul.f16x2 r7215, r7212, r7058;
}
{
sub.f16x2 r7218, r5728, r5744;
}
{
mul.f16x2 r7221, r7218, r7061;
}
{
add.f16x2 r7224, r7215, r7221;
}
{
add.f16x2 r7227, r7209, r7224;
}
st.local.u32 [rd4+148], r7227;
{
add.f16x2 r7230, r5712, r5760;
}
{
mul.f16x2 r7233, r7230, r7052;
}
{
add.f16x2 r7236, r4097, r7233;
}
{
add.f16x2 r7239, r5728, r5744;
}
{
mul.f16x2 r7242, r7239, r7056;
}
{
add.f16x2 r7245, r7236, r7242;
}
{
sub.f16x2 r7248, r5706, r5754;
}
{
mul.f16x2 r7251, r7248, r7054;
}
{
sub.f16x2 r7254, r5722, r5738;
}
{
mul.f16x2 r7257, r7254, r7058;
}
{
add.f16x2 r7260, r7251, r7257;
}
{
add.f16x2 r7263, r7245, r7260;
}
st.local.u32 [rd4+72], r7263;
{
add.f16x2 r7266, r5712, r5760;
}
{
mul.f16x2 r7269, r7266, r7052;
}
{
add.f16x2 r7272, r4097, r7269;
}
{
add.f16x2 r7275, r5728, r5744;
}
{
mul.f16x2 r7278, r7275, r7056;
}
{
add.f16x2 r7281, r7272, r7278;
}
{
sub.f16x2 r7284, r5706, r5754;
}
{
mul.f16x2 r7287, r7284, r7054;
}
{
sub.f16x2 r7290, r5722, r5738;
}
{
mul.f16x2 r7293, r7290, r7058;
}
{
add.f16x2 r7296, r7287, r7293;
}
{
sub.f16x2 r7299, r7281, r7296;
}
st.local.u32 [rd4+192], r7299;
{
add.f16x2 r7302, r5712, r5760;
}
{
mul.f16x2 r7305, r7302, r7056;
}
{
add.f16x2 r7308, r4097, r7305;
}
{
add.f16x2 r7311, r5728, r5744;
}
{
mul.f16x2 r7314, r7311, r7060;
}
{
add.f16x2 r7317, r7308, r7314;
}
{
sub.f16x2 r7320, r5706, r5754;
}
{
mul.f16x2 r7323, r7320, r7058;
}
{
sub.f16x2 r7326, r5722, r5738;
}
{
mul.f16x2 r7329, r7326, r7061;
}
{
add.f16x2 r7332, r7323, r7329;
}
{
add.f16x2 r7335, r7317, r7332;
}
st.local.u32 [rd4+112], r7335;
{
add.f16x2 r7338, r5712, r5760;
}
{
mul.f16x2 r7341, r7338, r7056;
}
{
add.f16x2 r7344, r4097, r7341;
}
{
add.f16x2 r7347, r5728, r5744;
}
{
mul.f16x2 r7350, r7347, r7060;
}
{
add.f16x2 r7353, r7344, r7350;
}
{
sub.f16x2 r7356, r5706, r5754;
}
{
mul.f16x2 r7359, r7356, r7058;
}
{
sub.f16x2 r7362, r5722, r5738;
}
{
mul.f16x2 r7365, r7362, r7061;
}
{
add.f16x2 r7368, r7359, r7365;
}
{
sub.f16x2 r7371, r7353, r7368;
}
st.local.u32 [rd4+152], r7371;
mul.wide.u32 rd13, r3, 1374389535;
shr.u64 rd14, rd13, 35;
cvt.u32.u64 r15, rd14;
cvt.rn.f32.u32 f460, r15;
mul.f32 f461, f460, 0f3C24B5BE;
cos.approx.f32 f456, f461;
sin.approx.f32 f462, f461;
neg.f32 f457, f462;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f456;
cvt.rn.f16.f32 high, f457;
mov.b32 r11313, {low, high};
}
mov.u32 r11312, 1;
{
.reg .f16 low, high;
mov.b32 {low, high}, r11313;
mov.b32 r7452, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11313;
mov.b32 r7454, {high, high};
}
bra.uni LBB1_4;
LBB1_5:
ld.local.u32 r11311, [rd7+60];
LBB1_4:
{
.reg .f16 low, high;
mov.b32 {low, high}, r11313;
mov.b32 r7432, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11313;
mov.b32 r7434, {high, high};
}
mul.wide.u32 rd15, r11312, 8;
add.s64 rd16, rd3, rd15;
add.s64 rd7, rd16, 4;
ld.local.u32 r7437, [rd16+4];
{
mul.f16x2 r7436, r7437, r7434;
}
{
fma.rn.f16x2 r7439, r11311, r7432, r7436;
}
st.local.u32 [rd16], r7439;
{
mul.f16x2 r7443, r11311, r7434;
}
{
neg.f16x2 r7446, r7443;
}
{
fma.rn.f16x2 r7448, r7437, r7432, r7446;
}
st.local.u32 [rd16+4], r7448;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7456, {low, high};
}
{
mul.f16x2 r7457, r7454, r7456;
}
{
mul.f16x2 r7460, r11313, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r11313;
mov.b32 r7463, {high, low};
}
{
fma.rn.f16x2 r7465, r7457, r7463, r7460;
}
ld.local.u32 r7481, [rd16+8];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7465;
mov.b32 r7469, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7465;
mov.b32 r7471, {high, high};
}
ld.local.u32 r7486, [rd16+12];
{
mul.f16x2 r7473, r7486, r7471;
}
{
fma.rn.f16x2 r7476, r7481, r7469, r7473;
}
st.local.u32 [rd16+8], r7476;
{
mul.f16x2 r7480, r7481, r7471;
}
{
neg.f16x2 r7483, r7480;
}
{
fma.rn.f16x2 r7485, r7486, r7469, r7483;
}
st.local.u32 [rd16+12], r7485;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7493, {low, high};
}
{
mul.f16x2 r7494, r7454, r7493;
}
{
mul.f16x2 r7497, r7465, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7465;
mov.b32 r7500, {high, low};
}
{
fma.rn.f16x2 r7502, r7494, r7500, r7497;
}
ld.local.u32 r7518, [rd16+16];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7502;
mov.b32 r7506, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7502;
mov.b32 r7508, {high, high};
}
ld.local.u32 r7523, [rd16+20];
{
mul.f16x2 r7510, r7523, r7508;
}
{
fma.rn.f16x2 r7513, r7518, r7506, r7510;
}
st.local.u32 [rd16+16], r7513;
{
mul.f16x2 r7517, r7518, r7508;
}
{
neg.f16x2 r7520, r7517;
}
{
fma.rn.f16x2 r7522, r7523, r7506, r7520;
}
st.local.u32 [rd16+20], r7522;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7530, {low, high};
}
{
mul.f16x2 r7531, r7454, r7530;
}
{
mul.f16x2 r7534, r7502, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7502;
mov.b32 r7537, {high, low};
}
{
fma.rn.f16x2 r7539, r7531, r7537, r7534;
}
ld.local.u32 r7555, [rd16+24];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7539;
mov.b32 r7543, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7539;
mov.b32 r7545, {high, high};
}
ld.local.u32 r7560, [rd16+28];
{
mul.f16x2 r7547, r7560, r7545;
}
{
fma.rn.f16x2 r7550, r7555, r7543, r7547;
}
st.local.u32 [rd16+24], r7550;
{
mul.f16x2 r7554, r7555, r7545;
}
{
neg.f16x2 r7557, r7554;
}
{
fma.rn.f16x2 r7559, r7560, r7543, r7557;
}
st.local.u32 [rd16+28], r7559;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7567, {low, high};
}
{
mul.f16x2 r7568, r7454, r7567;
}
{
mul.f16x2 r7571, r7539, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7539;
mov.b32 r7574, {high, low};
}
{
fma.rn.f16x2 r7576, r7568, r7574, r7571;
}
ld.local.u32 r7588, [rd16+32];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7576;
mov.b32 r7580, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7576;
mov.b32 r7582, {high, high};
}
ld.local.u32 r7597, [rd16+36];
{
mul.f16x2 r7584, r7597, r7582;
}
{
fma.rn.f16x2 r7587, r7588, r7580, r7584;
}
st.local.u32 [rd16+32], r7587;
{
mul.f16x2 r7591, r7588, r7582;
}
{
neg.f16x2 r7594, r7591;
}
{
fma.rn.f16x2 r7596, r7597, r7580, r7594;
}
st.local.u32 [rd16+36], r7596;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7604, {low, high};
}
{
mul.f16x2 r7605, r7454, r7604;
}
{
mul.f16x2 r7608, r7576, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7576;
mov.b32 r7611, {high, low};
}
{
fma.rn.f16x2 r7613, r7605, r7611, r7608;
}
ld.local.u32 r7625, [rd16+40];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7613;
mov.b32 r7617, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7613;
mov.b32 r7619, {high, high};
}
ld.local.u32 r7622, [rd16+44];
{
mul.f16x2 r7621, r7622, r7619;
}
{
fma.rn.f16x2 r7624, r7625, r7617, r7621;
}
st.local.u32 [rd16+40], r7624;
{
mul.f16x2 r7628, r7625, r7619;
}
{
neg.f16x2 r7631, r7628;
}
{
fma.rn.f16x2 r7633, r7622, r7617, r7631;
}
st.local.u32 [rd16+44], r7633;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7641, {low, high};
}
{
mul.f16x2 r7642, r7454, r7641;
}
{
mul.f16x2 r7645, r7613, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7613;
mov.b32 r7648, {high, low};
}
{
fma.rn.f16x2 r7650, r7642, r7648, r7645;
}
ld.local.u32 r7662, [rd16+48];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7650;
mov.b32 r7654, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7650;
mov.b32 r7656, {high, high};
}
ld.local.u32 r7659, [rd16+52];
{
mul.f16x2 r7658, r7659, r7656;
}
{
fma.rn.f16x2 r7661, r7662, r7654, r7658;
}
st.local.u32 [rd16+48], r7661;
{
mul.f16x2 r7665, r7662, r7656;
}
{
neg.f16x2 r7668, r7665;
}
{
fma.rn.f16x2 r7670, r7659, r7654, r7668;
}
st.local.u32 [rd16+52], r7670;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7678, {low, high};
}
{
mul.f16x2 r7679, r7454, r7678;
}
{
mul.f16x2 r7682, r7650, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7650;
mov.b32 r7685, {high, low};
}
{
fma.rn.f16x2 r7687, r7679, r7685, r7682;
}
ld.local.u32 r7699, [rd16+56];
{
.reg .f16 low, high;
mov.b32 {low, high}, r7687;
mov.b32 r7691, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7687;
mov.b32 r7693, {high, high};
}
ld.local.u32 r7696, [rd16+60];
{
mul.f16x2 r7695, r7696, r7693;
}
{
fma.rn.f16x2 r7698, r7699, r7691, r7695;
}
st.local.u32 [rd16+56], r7698;
{
mul.f16x2 r7702, r7699, r7693;
}
{
neg.f16x2 r7705, r7702;
}
{
fma.rn.f16x2 r7707, r7696, r7691, r7705;
}
st.local.u32 [rd16+60], r7707;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f238;
cvt.rn.f16.f32 high, f239;
mov.b32 r7715, {low, high};
}
{
mul.f16x2 r7716, r7454, r7715;
}
{
mul.f16x2 r7719, r7687, r7452;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7687;
mov.b32 r7722, {high, low};
}
{
fma.rn.f16x2 r11313, r7716, r7722, r7719;
}
add.s32 r11312, r11312, 8;
setp.eq.s32 p2, r11312, 25;
@p2 bra LBB1_6;
bra.uni LBB1_5;
LBB1_6:
mul.lo.s32 r11253, r15, 25;
sub.s32 r11254, r3, r11253;
shl.b32 r11255, r11254, 2;
add.s32 r11256, r12, r11255;
barrier.sync 0;
mad.lo.s32 r11257, r15, 2500, r11256;
ld.local.u32 r11258, [rd3];
st.shared.u32 [r11257], r11258;
ld.local.u32 r11259, [rd4+4];
st.shared.u32 [r11257+100], r11259;
ld.local.u32 r11260, [rd4+12];
st.shared.u32 [r11257+200], r11260;
ld.local.u32 r11261, [rd4+20];
st.shared.u32 [r11257+300], r11261;
ld.local.u32 r11262, [rd4+28];
st.shared.u32 [r11257+400], r11262;
ld.local.u32 r11263, [rd4+36];
st.shared.u32 [r11257+500], r11263;
ld.local.u32 r11264, [rd4+44];
st.shared.u32 [r11257+600], r11264;
ld.local.u32 r11265, [rd4+52];
st.shared.u32 [r11257+700], r11265;
ld.local.u32 r11266, [rd4+60];
st.shared.u32 [r11257+800], r11266;
ld.local.u32 r11267, [rd4+68];
st.shared.u32 [r11257+900], r11267;
ld.local.u32 r11268, [rd4+76];
st.shared.u32 [r11257+1000], r11268;
ld.local.u32 r11269, [rd4+84];
st.shared.u32 [r11257+1100], r11269;
ld.local.u32 r11270, [rd4+92];
st.shared.u32 [r11257+1200], r11270;
ld.local.u32 r11271, [rd4+100];
st.shared.u32 [r11257+1300], r11271;
ld.local.u32 r11272, [rd4+108];
st.shared.u32 [r11257+1400], r11272;
ld.local.u32 r11273, [rd4+116];
st.shared.u32 [r11257+1500], r11273;
ld.local.u32 r11274, [rd4+124];
st.shared.u32 [r11257+1600], r11274;
ld.local.u32 r11275, [rd4+132];
st.shared.u32 [r11257+1700], r11275;
ld.local.u32 r11276, [rd4+140];
st.shared.u32 [r11257+1800], r11276;
ld.local.u32 r11277, [rd4+148];
st.shared.u32 [r11257+1900], r11277;
ld.local.u32 r11278, [rd4+156];
st.shared.u32 [r11257+2000], r11278;
ld.local.u32 r11279, [rd4+164];
st.shared.u32 [r11257+2100], r11279;
ld.local.u32 r11280, [rd4+172];
st.shared.u32 [r11257+2200], r11280;
ld.local.u32 r11281, [rd4+180];
st.shared.u32 [r11257+2300], r11281;
ld.local.u32 r11282, [rd4+188];
st.shared.u32 [r11257+2400], r11282;
barrier.sync 0;
ld.shared.u32 r7742, [r13];
ld.shared.u32 r8064, [r13+2500];
ld.shared.u32 r8386, [r13+5000];
ld.shared.u32 r8708, [r13+7500];
ld.shared.u32 r9030, [r13+10000];
ld.shared.u32 r7739, [r13+12500];
ld.shared.u32 r8061, [r13+15000];
ld.shared.u32 r8383, [r13+17500];
ld.shared.u32 r8705, [r13+20000];
ld.shared.u32 r9027, [r13+22500];
ld.shared.u32 r7745, [r13+25000];
ld.shared.u32 r8067, [r13+27500];
ld.shared.u32 r8389, [r13+30000];
ld.shared.u32 r8711, [r13+32500];
ld.shared.u32 r9033, [r13+35000];
ld.shared.u32 r7746, [r13+37500];
ld.shared.u32 r8068, [r13+40000];
ld.shared.u32 r8390, [r13+42500];
ld.shared.u32 r8712, [r13+45000];
ld.shared.u32 r9034, [r13+47500];
ld.shared.u32 r7740, [r13+50000];
ld.shared.u32 r8062, [r13+52500];
ld.shared.u32 r8384, [r13+55000];
ld.shared.u32 r8706, [r13+57500];
ld.shared.u32 r9028, [r13+60000];
barrier.sync 0;
ld.local.u32 r11283, [rd4];
st.shared.u32 [r11257], r11283;
ld.local.u32 r11284, [rd4+8];
st.shared.u32 [r11257+100], r11284;
ld.local.u32 r11285, [rd4+16];
st.shared.u32 [r11257+200], r11285;
ld.local.u32 r11286, [rd4+24];
st.shared.u32 [r11257+300], r11286;
ld.local.u32 r11287, [rd4+32];
st.shared.u32 [r11257+400], r11287;
ld.local.u32 r11288, [rd4+40];
st.shared.u32 [r11257+500], r11288;
ld.local.u32 r11289, [rd4+48];
st.shared.u32 [r11257+600], r11289;
ld.local.u32 r11290, [rd4+56];
st.shared.u32 [r11257+700], r11290;
ld.local.u32 r11291, [rd4+64];
st.shared.u32 [r11257+800], r11291;
ld.local.u32 r11292, [rd4+72];
st.shared.u32 [r11257+900], r11292;
ld.local.u32 r11293, [rd4+80];
st.shared.u32 [r11257+1000], r11293;
ld.local.u32 r11294, [rd4+88];
st.shared.u32 [r11257+1100], r11294;
ld.local.u32 r11295, [rd4+96];
st.shared.u32 [r11257+1200], r11295;
ld.local.u32 r11296, [rd4+104];
st.shared.u32 [r11257+1300], r11296;
ld.local.u32 r11297, [rd4+112];
st.shared.u32 [r11257+1400], r11297;
ld.local.u32 r11298, [rd4+120];
st.shared.u32 [r11257+1500], r11298;
ld.local.u32 r11299, [rd4+128];
st.shared.u32 [r11257+1600], r11299;
ld.local.u32 r11300, [rd4+136];
st.shared.u32 [r11257+1700], r11300;
ld.local.u32 r11301, [rd4+144];
st.shared.u32 [r11257+1800], r11301;
ld.local.u32 r11302, [rd4+152];
st.shared.u32 [r11257+1900], r11302;
ld.local.u32 r11303, [rd4+160];
st.shared.u32 [r11257+2000], r11303;
ld.local.u32 r11304, [rd4+168];
st.shared.u32 [r11257+2100], r11304;
ld.local.u32 r11305, [rd4+176];
st.shared.u32 [r11257+2200], r11305;
ld.local.u32 r11306, [rd4+184];
st.shared.u32 [r11257+2300], r11306;
ld.local.u32 r11307, [rd4+192];
st.shared.u32 [r11257+2400], r11307;
barrier.sync 0;
ld.shared.u32 r7754, [r13];
ld.shared.u32 r8076, [r13+2500];
ld.shared.u32 r8398, [r13+5000];
ld.shared.u32 r8720, [r13+7500];
ld.shared.u32 r9042, [r13+10000];
ld.shared.u32 r7751, [r13+12500];
ld.shared.u32 r8073, [r13+15000];
ld.shared.u32 r8395, [r13+17500];
ld.shared.u32 r8717, [r13+20000];
ld.shared.u32 r9039, [r13+22500];
ld.shared.u32 r7757, [r13+25000];
ld.shared.u32 r8079, [r13+27500];
ld.shared.u32 r8401, [r13+30000];
ld.shared.u32 r8723, [r13+32500];
ld.shared.u32 r9045, [r13+35000];
ld.shared.u32 r7758, [r13+37500];
ld.shared.u32 r8080, [r13+40000];
ld.shared.u32 r8402, [r13+42500];
ld.shared.u32 r8724, [r13+45000];
ld.shared.u32 r9046, [r13+47500];
ld.shared.u32 r7752, [r13+50000];
ld.shared.u32 r8074, [r13+52500];
ld.shared.u32 r8396, [r13+55000];
ld.shared.u32 r8718, [r13+57500];
ld.shared.u32 r9040, [r13+60000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7728, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7729, {low, high};
}
{
neg.f16x2 r7730, r7729;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r7732, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r7733, {low, high};
}
{
neg.f16x2 r7734, r7733;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r7736, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r7737, {low, high};
}
{
add.f16x2 r7738, r7739, r7740;
}
{
add.f16x2 r7741, r7742, r7738;
}
{
add.f16x2 r7744, r7745, r7746;
}
{
add.f16x2 r7747, r7741, r7744;
}
{
add.f16x2 r7750, r7751, r7752;
}
{
add.f16x2 r7753, r7754, r7750;
}
{
add.f16x2 r7756, r7757, r7758;
}
{
add.f16x2 r7759, r7753, r7756;
}
{
add.f16x2 r7762, r7739, r7740;
}
{
mul.f16x2 r7765, r7762, r7728;
}
{
add.f16x2 r7768, r7742, r7765;
}
{
add.f16x2 r7771, r7745, r7746;
}
{
mul.f16x2 r7774, r7771, r7732;
}
{
add.f16x2 r7777, r7768, r7774;
}
{
sub.f16x2 r7780, r7751, r7752;
}
{
mul.f16x2 r7783, r7780, r7730;
}
{
sub.f16x2 r7786, r7757, r7758;
}
{
mul.f16x2 r7789, r7786, r7734;
}
{
add.f16x2 r7792, r7783, r7789;
}
{
sub.f16x2 r7795, r7777, r7792;
}
{
add.f16x2 r7798, r7739, r7740;
}
{
mul.f16x2 r7801, r7798, r7728;
}
{
add.f16x2 r7804, r7742, r7801;
}
{
add.f16x2 r7807, r7745, r7746;
}
{
mul.f16x2 r7810, r7807, r7732;
}
{
add.f16x2 r7813, r7804, r7810;
}
{
sub.f16x2 r7816, r7751, r7752;
}
{
mul.f16x2 r7819, r7816, r7730;
}
{
sub.f16x2 r7822, r7757, r7758;
}
{
mul.f16x2 r7825, r7822, r7734;
}
{
add.f16x2 r7828, r7819, r7825;
}
{
add.f16x2 r7831, r7813, r7828;
}
{
add.f16x2 r7834, r7739, r7740;
}
{
mul.f16x2 r7837, r7834, r7732;
}
{
add.f16x2 r7840, r7742, r7837;
}
{
add.f16x2 r7843, r7745, r7746;
}
{
mul.f16x2 r7846, r7843, r7736;
}
{
add.f16x2 r7849, r7840, r7846;
}
{
sub.f16x2 r7852, r7751, r7752;
}
{
mul.f16x2 r7855, r7852, r7734;
}
{
sub.f16x2 r7858, r7757, r7758;
}
{
mul.f16x2 r7861, r7858, r7737;
}
{
add.f16x2 r7864, r7855, r7861;
}
{
sub.f16x2 r7867, r7849, r7864;
}
{
add.f16x2 r7870, r7739, r7740;
}
{
mul.f16x2 r7873, r7870, r7732;
}
{
add.f16x2 r7876, r7742, r7873;
}
{
add.f16x2 r7879, r7745, r7746;
}
{
mul.f16x2 r7882, r7879, r7736;
}
{
add.f16x2 r7885, r7876, r7882;
}
{
sub.f16x2 r7888, r7751, r7752;
}
{
mul.f16x2 r7891, r7888, r7734;
}
{
sub.f16x2 r7894, r7757, r7758;
}
{
mul.f16x2 r7897, r7894, r7737;
}
{
add.f16x2 r7900, r7891, r7897;
}
{
add.f16x2 r7903, r7885, r7900;
}
{
add.f16x2 r7906, r7751, r7752;
}
{
mul.f16x2 r7909, r7906, r7728;
}
{
add.f16x2 r7912, r7754, r7909;
}
{
add.f16x2 r7915, r7757, r7758;
}
{
mul.f16x2 r7918, r7915, r7732;
}
{
add.f16x2 r7921, r7912, r7918;
}
{
sub.f16x2 r7924, r7739, r7740;
}
{
mul.f16x2 r7927, r7924, r7730;
}
{
sub.f16x2 r7930, r7745, r7746;
}
{
mul.f16x2 r7933, r7930, r7734;
}
{
add.f16x2 r7936, r7927, r7933;
}
{
add.f16x2 r7939, r7921, r7936;
}
{
add.f16x2 r7942, r7751, r7752;
}
{
mul.f16x2 r7945, r7942, r7728;
}
{
add.f16x2 r7948, r7754, r7945;
}
{
add.f16x2 r7951, r7757, r7758;
}
{
mul.f16x2 r7954, r7951, r7732;
}
{
add.f16x2 r7957, r7948, r7954;
}
{
sub.f16x2 r7960, r7739, r7740;
}
{
mul.f16x2 r7963, r7960, r7730;
}
{
sub.f16x2 r7966, r7745, r7746;
}
{
mul.f16x2 r7969, r7966, r7734;
}
{
add.f16x2 r7972, r7963, r7969;
}
{
sub.f16x2 r7975, r7957, r7972;
}
{
add.f16x2 r7978, r7751, r7752;
}
{
mul.f16x2 r7981, r7978, r7732;
}
{
add.f16x2 r7984, r7754, r7981;
}
{
add.f16x2 r7987, r7757, r7758;
}
{
mul.f16x2 r7990, r7987, r7736;
}
{
add.f16x2 r7993, r7984, r7990;
}
{
sub.f16x2 r7996, r7739, r7740;
}
{
mul.f16x2 r7999, r7996, r7734;
}
{
sub.f16x2 r8002, r7745, r7746;
}
{
mul.f16x2 r8005, r8002, r7737;
}
{
add.f16x2 r8008, r7999, r8005;
}
{
add.f16x2 r8011, r7993, r8008;
}
{
add.f16x2 r8014, r7751, r7752;
}
{
mul.f16x2 r8017, r8014, r7732;
}
{
add.f16x2 r8020, r7754, r8017;
}
{
add.f16x2 r8023, r7757, r7758;
}
{
mul.f16x2 r8026, r8023, r7736;
}
{
add.f16x2 r8029, r8020, r8026;
}
{
sub.f16x2 r8032, r7739, r7740;
}
{
mul.f16x2 r8035, r8032, r7734;
}
{
sub.f16x2 r8038, r7745, r7746;
}
{
mul.f16x2 r8041, r8038, r7737;
}
{
add.f16x2 r8044, r8035, r8041;
}
{
sub.f16x2 r8047, r8029, r8044;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8050, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8051, {low, high};
}
{
neg.f16x2 r8052, r8051;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8054, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8055, {low, high};
}
{
neg.f16x2 r8056, r8055;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8058, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8059, {low, high};
}
{
add.f16x2 r8060, r8061, r8062;
}
{
add.f16x2 r8063, r8064, r8060;
}
{
add.f16x2 r8066, r8067, r8068;
}
{
add.f16x2 r8069, r8063, r8066;
}
{
add.f16x2 r8072, r8073, r8074;
}
{
add.f16x2 r8075, r8076, r8072;
}
{
add.f16x2 r8078, r8079, r8080;
}
{
add.f16x2 r8081, r8075, r8078;
}
{
add.f16x2 r8084, r8061, r8062;
}
{
mul.f16x2 r8087, r8084, r8050;
}
{
add.f16x2 r8090, r8064, r8087;
}
{
add.f16x2 r8093, r8067, r8068;
}
{
mul.f16x2 r8096, r8093, r8054;
}
{
add.f16x2 r8099, r8090, r8096;
}
{
sub.f16x2 r8102, r8073, r8074;
}
{
mul.f16x2 r8105, r8102, r8052;
}
{
sub.f16x2 r8108, r8079, r8080;
}
{
mul.f16x2 r8111, r8108, r8056;
}
{
add.f16x2 r8114, r8105, r8111;
}
{
sub.f16x2 r8117, r8099, r8114;
}
{
add.f16x2 r8120, r8061, r8062;
}
{
mul.f16x2 r8123, r8120, r8050;
}
{
add.f16x2 r8126, r8064, r8123;
}
{
add.f16x2 r8129, r8067, r8068;
}
{
mul.f16x2 r8132, r8129, r8054;
}
{
add.f16x2 r8135, r8126, r8132;
}
{
sub.f16x2 r8138, r8073, r8074;
}
{
mul.f16x2 r8141, r8138, r8052;
}
{
sub.f16x2 r8144, r8079, r8080;
}
{
mul.f16x2 r8147, r8144, r8056;
}
{
add.f16x2 r8150, r8141, r8147;
}
{
add.f16x2 r8153, r8135, r8150;
}
{
add.f16x2 r8156, r8061, r8062;
}
{
mul.f16x2 r8159, r8156, r8054;
}
{
add.f16x2 r8162, r8064, r8159;
}
{
add.f16x2 r8165, r8067, r8068;
}
{
mul.f16x2 r8168, r8165, r8058;
}
{
add.f16x2 r8171, r8162, r8168;
}
{
sub.f16x2 r8174, r8073, r8074;
}
{
mul.f16x2 r8177, r8174, r8056;
}
{
sub.f16x2 r8180, r8079, r8080;
}
{
mul.f16x2 r8183, r8180, r8059;
}
{
add.f16x2 r8186, r8177, r8183;
}
{
sub.f16x2 r8189, r8171, r8186;
}
{
add.f16x2 r8192, r8061, r8062;
}
{
mul.f16x2 r8195, r8192, r8054;
}
{
add.f16x2 r8198, r8064, r8195;
}
{
add.f16x2 r8201, r8067, r8068;
}
{
mul.f16x2 r8204, r8201, r8058;
}
{
add.f16x2 r8207, r8198, r8204;
}
{
sub.f16x2 r8210, r8073, r8074;
}
{
mul.f16x2 r8213, r8210, r8056;
}
{
sub.f16x2 r8216, r8079, r8080;
}
{
mul.f16x2 r8219, r8216, r8059;
}
{
add.f16x2 r8222, r8213, r8219;
}
{
add.f16x2 r8225, r8207, r8222;
}
{
add.f16x2 r8228, r8073, r8074;
}
{
mul.f16x2 r8231, r8228, r8050;
}
{
add.f16x2 r8234, r8076, r8231;
}
{
add.f16x2 r8237, r8079, r8080;
}
{
mul.f16x2 r8240, r8237, r8054;
}
{
add.f16x2 r8243, r8234, r8240;
}
{
sub.f16x2 r8246, r8061, r8062;
}
{
mul.f16x2 r8249, r8246, r8052;
}
{
sub.f16x2 r8252, r8067, r8068;
}
{
mul.f16x2 r8255, r8252, r8056;
}
{
add.f16x2 r8258, r8249, r8255;
}
{
add.f16x2 r8261, r8243, r8258;
}
{
add.f16x2 r8264, r8073, r8074;
}
{
mul.f16x2 r8267, r8264, r8050;
}
{
add.f16x2 r8270, r8076, r8267;
}
{
add.f16x2 r8273, r8079, r8080;
}
{
mul.f16x2 r8276, r8273, r8054;
}
{
add.f16x2 r8279, r8270, r8276;
}
{
sub.f16x2 r8282, r8061, r8062;
}
{
mul.f16x2 r8285, r8282, r8052;
}
{
sub.f16x2 r8288, r8067, r8068;
}
{
mul.f16x2 r8291, r8288, r8056;
}
{
add.f16x2 r8294, r8285, r8291;
}
{
sub.f16x2 r8297, r8279, r8294;
}
{
add.f16x2 r8300, r8073, r8074;
}
{
mul.f16x2 r8303, r8300, r8054;
}
{
add.f16x2 r8306, r8076, r8303;
}
{
add.f16x2 r8309, r8079, r8080;
}
{
mul.f16x2 r8312, r8309, r8058;
}
{
add.f16x2 r8315, r8306, r8312;
}
{
sub.f16x2 r8318, r8061, r8062;
}
{
mul.f16x2 r8321, r8318, r8056;
}
{
sub.f16x2 r8324, r8067, r8068;
}
{
mul.f16x2 r8327, r8324, r8059;
}
{
add.f16x2 r8330, r8321, r8327;
}
{
add.f16x2 r8333, r8315, r8330;
}
{
add.f16x2 r8336, r8073, r8074;
}
{
mul.f16x2 r8339, r8336, r8054;
}
{
add.f16x2 r8342, r8076, r8339;
}
{
add.f16x2 r8345, r8079, r8080;
}
{
mul.f16x2 r8348, r8345, r8058;
}
{
add.f16x2 r8351, r8342, r8348;
}
{
sub.f16x2 r8354, r8061, r8062;
}
{
mul.f16x2 r8357, r8354, r8056;
}
{
sub.f16x2 r8360, r8067, r8068;
}
{
mul.f16x2 r8363, r8360, r8059;
}
{
add.f16x2 r8366, r8357, r8363;
}
{
sub.f16x2 r8369, r8351, r8366;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8372, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8373, {low, high};
}
{
neg.f16x2 r8374, r8373;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8376, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8377, {low, high};
}
{
neg.f16x2 r8378, r8377;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8380, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8381, {low, high};
}
{
add.f16x2 r8382, r8383, r8384;
}
{
add.f16x2 r8385, r8386, r8382;
}
{
add.f16x2 r8388, r8389, r8390;
}
{
add.f16x2 r8391, r8385, r8388;
}
{
add.f16x2 r8394, r8395, r8396;
}
{
add.f16x2 r8397, r8398, r8394;
}
{
add.f16x2 r8400, r8401, r8402;
}
{
add.f16x2 r8403, r8397, r8400;
}
{
add.f16x2 r8406, r8383, r8384;
}
{
mul.f16x2 r8409, r8406, r8372;
}
{
add.f16x2 r8412, r8386, r8409;
}
{
add.f16x2 r8415, r8389, r8390;
}
{
mul.f16x2 r8418, r8415, r8376;
}
{
add.f16x2 r8421, r8412, r8418;
}
{
sub.f16x2 r8424, r8395, r8396;
}
{
mul.f16x2 r8427, r8424, r8374;
}
{
sub.f16x2 r8430, r8401, r8402;
}
{
mul.f16x2 r8433, r8430, r8378;
}
{
add.f16x2 r8436, r8427, r8433;
}
{
sub.f16x2 r8439, r8421, r8436;
}
{
add.f16x2 r8442, r8383, r8384;
}
{
mul.f16x2 r8445, r8442, r8372;
}
{
add.f16x2 r8448, r8386, r8445;
}
{
add.f16x2 r8451, r8389, r8390;
}
{
mul.f16x2 r8454, r8451, r8376;
}
{
add.f16x2 r8457, r8448, r8454;
}
{
sub.f16x2 r8460, r8395, r8396;
}
{
mul.f16x2 r8463, r8460, r8374;
}
{
sub.f16x2 r8466, r8401, r8402;
}
{
mul.f16x2 r8469, r8466, r8378;
}
{
add.f16x2 r8472, r8463, r8469;
}
{
add.f16x2 r8475, r8457, r8472;
}
{
add.f16x2 r8478, r8383, r8384;
}
{
mul.f16x2 r8481, r8478, r8376;
}
{
add.f16x2 r8484, r8386, r8481;
}
{
add.f16x2 r8487, r8389, r8390;
}
{
mul.f16x2 r8490, r8487, r8380;
}
{
add.f16x2 r8493, r8484, r8490;
}
{
sub.f16x2 r8496, r8395, r8396;
}
{
mul.f16x2 r8499, r8496, r8378;
}
{
sub.f16x2 r8502, r8401, r8402;
}
{
mul.f16x2 r8505, r8502, r8381;
}
{
add.f16x2 r8508, r8499, r8505;
}
{
sub.f16x2 r8511, r8493, r8508;
}
{
add.f16x2 r8514, r8383, r8384;
}
{
mul.f16x2 r8517, r8514, r8376;
}
{
add.f16x2 r8520, r8386, r8517;
}
{
add.f16x2 r8523, r8389, r8390;
}
{
mul.f16x2 r8526, r8523, r8380;
}
{
add.f16x2 r8529, r8520, r8526;
}
{
sub.f16x2 r8532, r8395, r8396;
}
{
mul.f16x2 r8535, r8532, r8378;
}
{
sub.f16x2 r8538, r8401, r8402;
}
{
mul.f16x2 r8541, r8538, r8381;
}
{
add.f16x2 r8544, r8535, r8541;
}
{
add.f16x2 r8547, r8529, r8544;
}
{
add.f16x2 r8550, r8395, r8396;
}
{
mul.f16x2 r8553, r8550, r8372;
}
{
add.f16x2 r8556, r8398, r8553;
}
{
add.f16x2 r8559, r8401, r8402;
}
{
mul.f16x2 r8562, r8559, r8376;
}
{
add.f16x2 r8565, r8556, r8562;
}
{
sub.f16x2 r8568, r8383, r8384;
}
{
mul.f16x2 r8571, r8568, r8374;
}
{
sub.f16x2 r8574, r8389, r8390;
}
{
mul.f16x2 r8577, r8574, r8378;
}
{
add.f16x2 r8580, r8571, r8577;
}
{
add.f16x2 r8583, r8565, r8580;
}
{
add.f16x2 r8586, r8395, r8396;
}
{
mul.f16x2 r8589, r8586, r8372;
}
{
add.f16x2 r8592, r8398, r8589;
}
{
add.f16x2 r8595, r8401, r8402;
}
{
mul.f16x2 r8598, r8595, r8376;
}
{
add.f16x2 r8601, r8592, r8598;
}
{
sub.f16x2 r8604, r8383, r8384;
}
{
mul.f16x2 r8607, r8604, r8374;
}
{
sub.f16x2 r8610, r8389, r8390;
}
{
mul.f16x2 r8613, r8610, r8378;
}
{
add.f16x2 r8616, r8607, r8613;
}
{
sub.f16x2 r8619, r8601, r8616;
}
{
add.f16x2 r8622, r8395, r8396;
}
{
mul.f16x2 r8625, r8622, r8376;
}
{
add.f16x2 r8628, r8398, r8625;
}
{
add.f16x2 r8631, r8401, r8402;
}
{
mul.f16x2 r8634, r8631, r8380;
}
{
add.f16x2 r8637, r8628, r8634;
}
{
sub.f16x2 r8640, r8383, r8384;
}
{
mul.f16x2 r8643, r8640, r8378;
}
{
sub.f16x2 r8646, r8389, r8390;
}
{
mul.f16x2 r8649, r8646, r8381;
}
{
add.f16x2 r8652, r8643, r8649;
}
{
add.f16x2 r8655, r8637, r8652;
}
{
add.f16x2 r8658, r8395, r8396;
}
{
mul.f16x2 r8661, r8658, r8376;
}
{
add.f16x2 r8664, r8398, r8661;
}
{
add.f16x2 r8667, r8401, r8402;
}
{
mul.f16x2 r8670, r8667, r8380;
}
{
add.f16x2 r8673, r8664, r8670;
}
{
sub.f16x2 r8676, r8383, r8384;
}
{
mul.f16x2 r8679, r8676, r8378;
}
{
sub.f16x2 r8682, r8389, r8390;
}
{
mul.f16x2 r8685, r8682, r8381;
}
{
add.f16x2 r8688, r8679, r8685;
}
{
sub.f16x2 r8691, r8673, r8688;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8694, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8695, {low, high};
}
{
neg.f16x2 r8696, r8695;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r8698, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r8699, {low, high};
}
{
neg.f16x2 r8700, r8699;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r8702, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r8703, {low, high};
}
{
add.f16x2 r8704, r8705, r8706;
}
{
add.f16x2 r8707, r8708, r8704;
}
{
add.f16x2 r8710, r8711, r8712;
}
{
add.f16x2 r8713, r8707, r8710;
}
{
add.f16x2 r8716, r8717, r8718;
}
{
add.f16x2 r8719, r8720, r8716;
}
{
add.f16x2 r8722, r8723, r8724;
}
{
add.f16x2 r8725, r8719, r8722;
}
{
add.f16x2 r8728, r8705, r8706;
}
{
mul.f16x2 r8731, r8728, r8694;
}
{
add.f16x2 r8734, r8708, r8731;
}
{
add.f16x2 r8737, r8711, r8712;
}
{
mul.f16x2 r8740, r8737, r8698;
}
{
add.f16x2 r8743, r8734, r8740;
}
{
sub.f16x2 r8746, r8717, r8718;
}
{
mul.f16x2 r8749, r8746, r8696;
}
{
sub.f16x2 r8752, r8723, r8724;
}
{
mul.f16x2 r8755, r8752, r8700;
}
{
add.f16x2 r8758, r8749, r8755;
}
{
sub.f16x2 r8761, r8743, r8758;
}
{
add.f16x2 r8764, r8705, r8706;
}
{
mul.f16x2 r8767, r8764, r8694;
}
{
add.f16x2 r8770, r8708, r8767;
}
{
add.f16x2 r8773, r8711, r8712;
}
{
mul.f16x2 r8776, r8773, r8698;
}
{
add.f16x2 r8779, r8770, r8776;
}
{
sub.f16x2 r8782, r8717, r8718;
}
{
mul.f16x2 r8785, r8782, r8696;
}
{
sub.f16x2 r8788, r8723, r8724;
}
{
mul.f16x2 r8791, r8788, r8700;
}
{
add.f16x2 r8794, r8785, r8791;
}
{
add.f16x2 r8797, r8779, r8794;
}
{
add.f16x2 r8800, r8705, r8706;
}
{
mul.f16x2 r8803, r8800, r8698;
}
{
add.f16x2 r8806, r8708, r8803;
}
{
add.f16x2 r8809, r8711, r8712;
}
{
mul.f16x2 r8812, r8809, r8702;
}
{
add.f16x2 r8815, r8806, r8812;
}
{
sub.f16x2 r8818, r8717, r8718;
}
{
mul.f16x2 r8821, r8818, r8700;
}
{
sub.f16x2 r8824, r8723, r8724;
}
{
mul.f16x2 r8827, r8824, r8703;
}
{
add.f16x2 r8830, r8821, r8827;
}
{
sub.f16x2 r8833, r8815, r8830;
}
{
add.f16x2 r8836, r8705, r8706;
}
{
mul.f16x2 r8839, r8836, r8698;
}
{
add.f16x2 r8842, r8708, r8839;
}
{
add.f16x2 r8845, r8711, r8712;
}
{
mul.f16x2 r8848, r8845, r8702;
}
{
add.f16x2 r8851, r8842, r8848;
}
{
sub.f16x2 r8854, r8717, r8718;
}
{
mul.f16x2 r8857, r8854, r8700;
}
{
sub.f16x2 r8860, r8723, r8724;
}
{
mul.f16x2 r8863, r8860, r8703;
}
{
add.f16x2 r8866, r8857, r8863;
}
{
add.f16x2 r8869, r8851, r8866;
}
{
add.f16x2 r8872, r8717, r8718;
}
{
mul.f16x2 r8875, r8872, r8694;
}
{
add.f16x2 r8878, r8720, r8875;
}
{
add.f16x2 r8881, r8723, r8724;
}
{
mul.f16x2 r8884, r8881, r8698;
}
{
add.f16x2 r8887, r8878, r8884;
}
{
sub.f16x2 r8890, r8705, r8706;
}
{
mul.f16x2 r8893, r8890, r8696;
}
{
sub.f16x2 r8896, r8711, r8712;
}
{
mul.f16x2 r8899, r8896, r8700;
}
{
add.f16x2 r8902, r8893, r8899;
}
{
add.f16x2 r8905, r8887, r8902;
}
{
add.f16x2 r8908, r8717, r8718;
}
{
mul.f16x2 r8911, r8908, r8694;
}
{
add.f16x2 r8914, r8720, r8911;
}
{
add.f16x2 r8917, r8723, r8724;
}
{
mul.f16x2 r8920, r8917, r8698;
}
{
add.f16x2 r8923, r8914, r8920;
}
{
sub.f16x2 r8926, r8705, r8706;
}
{
mul.f16x2 r8929, r8926, r8696;
}
{
sub.f16x2 r8932, r8711, r8712;
}
{
mul.f16x2 r8935, r8932, r8700;
}
{
add.f16x2 r8938, r8929, r8935;
}
{
sub.f16x2 r8941, r8923, r8938;
}
{
add.f16x2 r8944, r8717, r8718;
}
{
mul.f16x2 r8947, r8944, r8698;
}
{
add.f16x2 r8950, r8720, r8947;
}
{
add.f16x2 r8953, r8723, r8724;
}
{
mul.f16x2 r8956, r8953, r8702;
}
{
add.f16x2 r8959, r8950, r8956;
}
{
sub.f16x2 r8962, r8705, r8706;
}
{
mul.f16x2 r8965, r8962, r8700;
}
{
sub.f16x2 r8968, r8711, r8712;
}
{
mul.f16x2 r8971, r8968, r8703;
}
{
add.f16x2 r8974, r8965, r8971;
}
{
add.f16x2 r8977, r8959, r8974;
}
{
add.f16x2 r8980, r8717, r8718;
}
{
mul.f16x2 r8983, r8980, r8698;
}
{
add.f16x2 r8986, r8720, r8983;
}
{
add.f16x2 r8989, r8723, r8724;
}
{
mul.f16x2 r8992, r8989, r8702;
}
{
add.f16x2 r8995, r8986, r8992;
}
{
sub.f16x2 r8998, r8705, r8706;
}
{
mul.f16x2 r9001, r8998, r8700;
}
{
sub.f16x2 r9004, r8711, r8712;
}
{
mul.f16x2 r9007, r9004, r8703;
}
{
add.f16x2 r9010, r9001, r9007;
}
{
sub.f16x2 r9013, r8995, r9010;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9016, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9017, {low, high};
}
{
neg.f16x2 r9018, r9017;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9020, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9021, {low, high};
}
{
neg.f16x2 r9022, r9021;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9024, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9025, {low, high};
}
{
add.f16x2 r9026, r9027, r9028;
}
{
add.f16x2 r9029, r9030, r9026;
}
{
add.f16x2 r9032, r9033, r9034;
}
{
add.f16x2 r9035, r9029, r9032;
}
{
add.f16x2 r9038, r9039, r9040;
}
{
add.f16x2 r9041, r9042, r9038;
}
{
add.f16x2 r9044, r9045, r9046;
}
{
add.f16x2 r9047, r9041, r9044;
}
{
add.f16x2 r9050, r9027, r9028;
}
{
mul.f16x2 r9053, r9050, r9016;
}
{
add.f16x2 r9056, r9030, r9053;
}
{
add.f16x2 r9059, r9033, r9034;
}
{
mul.f16x2 r9062, r9059, r9020;
}
{
add.f16x2 r9065, r9056, r9062;
}
{
sub.f16x2 r9068, r9039, r9040;
}
{
mul.f16x2 r9071, r9068, r9018;
}
{
sub.f16x2 r9074, r9045, r9046;
}
{
mul.f16x2 r9077, r9074, r9022;
}
{
add.f16x2 r9080, r9071, r9077;
}
{
sub.f16x2 r9083, r9065, r9080;
}
{
add.f16x2 r9086, r9027, r9028;
}
{
mul.f16x2 r9089, r9086, r9016;
}
{
add.f16x2 r9092, r9030, r9089;
}
{
add.f16x2 r9095, r9033, r9034;
}
{
mul.f16x2 r9098, r9095, r9020;
}
{
add.f16x2 r9101, r9092, r9098;
}
{
sub.f16x2 r9104, r9039, r9040;
}
{
mul.f16x2 r9107, r9104, r9018;
}
{
sub.f16x2 r9110, r9045, r9046;
}
{
mul.f16x2 r9113, r9110, r9022;
}
{
add.f16x2 r9116, r9107, r9113;
}
{
add.f16x2 r9119, r9101, r9116;
}
{
add.f16x2 r9122, r9027, r9028;
}
{
mul.f16x2 r9125, r9122, r9020;
}
{
add.f16x2 r9128, r9030, r9125;
}
{
add.f16x2 r9131, r9033, r9034;
}
{
mul.f16x2 r9134, r9131, r9024;
}
{
add.f16x2 r9137, r9128, r9134;
}
{
sub.f16x2 r9140, r9039, r9040;
}
{
mul.f16x2 r9143, r9140, r9022;
}
{
sub.f16x2 r9146, r9045, r9046;
}
{
mul.f16x2 r9149, r9146, r9025;
}
{
add.f16x2 r9152, r9143, r9149;
}
{
sub.f16x2 r9155, r9137, r9152;
}
{
add.f16x2 r9158, r9027, r9028;
}
{
mul.f16x2 r9161, r9158, r9020;
}
{
add.f16x2 r9164, r9030, r9161;
}
{
add.f16x2 r9167, r9033, r9034;
}
{
mul.f16x2 r9170, r9167, r9024;
}
{
add.f16x2 r9173, r9164, r9170;
}
{
sub.f16x2 r9176, r9039, r9040;
}
{
mul.f16x2 r9179, r9176, r9022;
}
{
sub.f16x2 r9182, r9045, r9046;
}
{
mul.f16x2 r9185, r9182, r9025;
}
{
add.f16x2 r9188, r9179, r9185;
}
{
add.f16x2 r9191, r9173, r9188;
}
{
add.f16x2 r9194, r9039, r9040;
}
{
mul.f16x2 r9197, r9194, r9016;
}
{
add.f16x2 r9200, r9042, r9197;
}
{
add.f16x2 r9203, r9045, r9046;
}
{
mul.f16x2 r9206, r9203, r9020;
}
{
add.f16x2 r9209, r9200, r9206;
}
{
sub.f16x2 r9212, r9027, r9028;
}
{
mul.f16x2 r9215, r9212, r9018;
}
{
sub.f16x2 r9218, r9033, r9034;
}
{
mul.f16x2 r9221, r9218, r9022;
}
{
add.f16x2 r9224, r9215, r9221;
}
{
add.f16x2 r9227, r9209, r9224;
}
{
add.f16x2 r9230, r9039, r9040;
}
{
mul.f16x2 r9233, r9230, r9016;
}
{
add.f16x2 r9236, r9042, r9233;
}
{
add.f16x2 r9239, r9045, r9046;
}
{
mul.f16x2 r9242, r9239, r9020;
}
{
add.f16x2 r9245, r9236, r9242;
}
{
sub.f16x2 r9248, r9027, r9028;
}
{
mul.f16x2 r9251, r9248, r9018;
}
{
sub.f16x2 r9254, r9033, r9034;
}
{
mul.f16x2 r9257, r9254, r9022;
}
{
add.f16x2 r9260, r9251, r9257;
}
{
sub.f16x2 r9263, r9245, r9260;
}
{
add.f16x2 r9266, r9039, r9040;
}
{
mul.f16x2 r9269, r9266, r9020;
}
{
add.f16x2 r9272, r9042, r9269;
}
{
add.f16x2 r9275, r9045, r9046;
}
{
mul.f16x2 r9278, r9275, r9024;
}
{
add.f16x2 r9281, r9272, r9278;
}
{
sub.f16x2 r9284, r9027, r9028;
}
{
mul.f16x2 r9287, r9284, r9022;
}
{
sub.f16x2 r9290, r9033, r9034;
}
{
mul.f16x2 r9293, r9290, r9025;
}
{
add.f16x2 r9296, r9287, r9293;
}
{
add.f16x2 r9299, r9281, r9296;
}
{
add.f16x2 r9302, r9039, r9040;
}
{
mul.f16x2 r9305, r9302, r9020;
}
{
add.f16x2 r9308, r9042, r9305;
}
{
add.f16x2 r9311, r9045, r9046;
}
{
mul.f16x2 r9314, r9311, r9024;
}
{
add.f16x2 r9317, r9308, r9314;
}
{
sub.f16x2 r9320, r9027, r9028;
}
{
mul.f16x2 r9323, r9320, r9022;
}
{
sub.f16x2 r9326, r9033, r9034;
}
{
mul.f16x2 r9329, r9326, r9025;
}
{
add.f16x2 r9332, r9323, r9329;
}
{
sub.f16x2 r9335, r9317, r9332;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f62;
cvt.rn.f16.f32 high, f62;
mov.b32 r9338, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f64;
cvt.rn.f16.f32 high, f64;
mov.b32 r9339, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f66;
cvt.rn.f16.f32 high, f66;
mov.b32 r9340, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f68;
cvt.rn.f16.f32 high, f68;
mov.b32 r9341, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f70;
cvt.rn.f16.f32 high, f70;
mov.b32 r9342, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f72;
cvt.rn.f16.f32 high, f72;
mov.b32 r9343, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r9344, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r9345, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r9348, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r9349, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f90;
cvt.rn.f16.f32 high, f90;
mov.b32 r9352, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f92;
cvt.rn.f16.f32 high, f92;
mov.b32 r9353, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9354, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f96;
cvt.rn.f16.f32 high, f96;
mov.b32 r9355, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r9360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r9361, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f122;
cvt.rn.f16.f32 high, f122;
mov.b32 r9368, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f124;
cvt.rn.f16.f32 high, f124;
mov.b32 r9369, {low, high};
}
{
mul.f16x2 r9386, r8117, r9338;
}
{
mul.f16x2 r9389, r8261, r9339;
}
{
sub.f16x2 r9392, r9386, r9389;
}
{
mul.f16x2 r9395, r8117, r9339;
}
{
fma.rn.f16x2 r9398, r8261, r9338, r9395;
}
{
mul.f16x2 r9402, r8439, r9340;
}
{
mul.f16x2 r9405, r8583, r9341;
}
{
sub.f16x2 r9408, r9402, r9405;
}
{
mul.f16x2 r9411, r8439, r9341;
}
{
fma.rn.f16x2 r9414, r8583, r9340, r9411;
}
{
mul.f16x2 r9418, r8761, r9342;
}
{
mul.f16x2 r9421, r8905, r9343;
}
{
sub.f16x2 r9424, r9418, r9421;
}
{
mul.f16x2 r9427, r8761, r9343;
}
{
fma.rn.f16x2 r9430, r8905, r9342, r9427;
}
{
mul.f16x2 r9434, r9083, r9344;
}
{
mul.f16x2 r9437, r9227, r9345;
}
{
sub.f16x2 r9440, r9434, r9437;
}
{
mul.f16x2 r9443, r9083, r9345;
}
{
fma.rn.f16x2 r9446, r9227, r9344, r9443;
}
{
mul.f16x2 r9450, r8189, r9340;
}
{
mul.f16x2 r9453, r8333, r9341;
}
{
sub.f16x2 r9456, r9450, r9453;
}
{
mul.f16x2 r9459, r8189, r9341;
}
{
fma.rn.f16x2 r9462, r8333, r9340, r9459;
}
{
mul.f16x2 r9466, r8511, r9344;
}
{
mul.f16x2 r9469, r8655, r9345;
}
{
sub.f16x2 r9472, r9466, r9469;
}
{
mul.f16x2 r9475, r8511, r9345;
}
{
fma.rn.f16x2 r9478, r8655, r9344, r9475;
}
{
mul.f16x2 r9482, r8833, r9348;
}
{
mul.f16x2 r9485, r8977, r9349;
}
{
sub.f16x2 r9488, r9482, r9485;
}
{
mul.f16x2 r9491, r8833, r9349;
}
{
fma.rn.f16x2 r9494, r8977, r9348, r9491;
}
{
mul.f16x2 r9498, r9155, r9352;
}
{
mul.f16x2 r9501, r9299, r9353;
}
{
sub.f16x2 r9504, r9498, r9501;
}
{
mul.f16x2 r9507, r9155, r9353;
}
{
fma.rn.f16x2 r9510, r9299, r9352, r9507;
}
{
mul.f16x2 r9514, r8225, r9342;
}
{
mul.f16x2 r9517, r8369, r9343;
}
{
sub.f16x2 r9520, r9514, r9517;
}
{
mul.f16x2 r9523, r8225, r9343;
}
{
fma.rn.f16x2 r9526, r8369, r9342, r9523;
}
{
mul.f16x2 r9530, r8547, r9348;
}
{
mul.f16x2 r9533, r8691, r9349;
}
{
sub.f16x2 r9536, r9530, r9533;
}
{
mul.f16x2 r9539, r8547, r9349;
}
{
fma.rn.f16x2 r9542, r8691, r9348, r9539;
}
{
mul.f16x2 r9546, r8869, r9354;
}
{
mul.f16x2 r9549, r9013, r9355;
}
{
sub.f16x2 r9552, r9546, r9549;
}
{
mul.f16x2 r9555, r8869, r9355;
}
{
fma.rn.f16x2 r9558, r9013, r9354, r9555;
}
{
mul.f16x2 r9562, r9191, r9360;
}
{
mul.f16x2 r9565, r9335, r9361;
}
{
sub.f16x2 r9568, r9562, r9565;
}
{
mul.f16x2 r9571, r9191, r9361;
}
{
fma.rn.f16x2 r9574, r9335, r9360, r9571;
}
{
mul.f16x2 r9578, r8153, r9344;
}
{
mul.f16x2 r9581, r8297, r9345;
}
{
sub.f16x2 r9584, r9578, r9581;
}
{
mul.f16x2 r9587, r8153, r9345;
}
{
fma.rn.f16x2 r9590, r8297, r9344, r9587;
}
{
mul.f16x2 r9594, r8475, r9352;
}
{
mul.f16x2 r9597, r8619, r9353;
}
{
sub.f16x2 r9600, r9594, r9597;
}
{
mul.f16x2 r9603, r8475, r9353;
}
{
fma.rn.f16x2 r9606, r8619, r9352, r9603;
}
{
mul.f16x2 r9610, r8797, r9360;
}
{
mul.f16x2 r9613, r8941, r9361;
}
{
sub.f16x2 r9616, r9610, r9613;
}
{
mul.f16x2 r9619, r8797, r9361;
}
{
fma.rn.f16x2 r9622, r8941, r9360, r9619;
}
{
mul.f16x2 r9626, r9119, r9368;
}
{
mul.f16x2 r9629, r9263, r9369;
}
{
sub.f16x2 r9632, r9626, r9629;
}
{
mul.f16x2 r9635, r9119, r9369;
}
{
fma.rn.f16x2 r9638, r9263, r9368, r9635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9643, {low, high};
}
{
neg.f16x2 r9644, r9643;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9646, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9647, {low, high};
}
{
neg.f16x2 r9648, r9647;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9650, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9651, {low, high};
}
{
add.f16x2 r9652, r8069, r9035;
}
{
add.f16x2 r9655, r7747, r9652;
}
{
add.f16x2 r9658, r8391, r8713;
}
{
add.f16x2 %0, r9655, r9658;
}
{
add.f16x2 r9664, r8081, r9047;
}
{
add.f16x2 r9667, r7759, r9664;
}
{
add.f16x2 r9670, r8403, r8725;
}
{
add.f16x2 %1, r9667, r9670;
}
{
add.f16x2 r9676, r8069, r9035;
}
{
mul.f16x2 r9679, r9676, r9642;
}
{
add.f16x2 r9682, r7747, r9679;
}
{
add.f16x2 r9685, r8391, r8713;
}
{
mul.f16x2 r9688, r9685, r9646;
}
{
add.f16x2 r9691, r9682, r9688;
}
{
sub.f16x2 r9694, r8081, r9047;
}
{
mul.f16x2 r9697, r9694, r9644;
}
{
sub.f16x2 r9700, r8403, r8725;
}
{
mul.f16x2 r9703, r9700, r9648;
}
{
add.f16x2 r9706, r9697, r9703;
}
{
sub.f16x2 %10, r9691, r9706;
}
{
add.f16x2 r9712, r8069, r9035;
}
{
mul.f16x2 r9715, r9712, r9642;
}
{
add.f16x2 r9718, r7747, r9715;
}
{
add.f16x2 r9721, r8391, r8713;
}
{
mul.f16x2 r9724, r9721, r9646;
}
{
add.f16x2 r9727, r9718, r9724;
}
{
sub.f16x2 r9730, r8081, r9047;
}
{
mul.f16x2 r9733, r9730, r9644;
}
{
sub.f16x2 r9736, r8403, r8725;
}
{
mul.f16x2 r9739, r9736, r9648;
}
{
add.f16x2 r9742, r9733, r9739;
}
{
add.f16x2 %40, r9727, r9742;
}
{
add.f16x2 r9748, r8069, r9035;
}
{
mul.f16x2 r9751, r9748, r9646;
}
{
add.f16x2 r9754, r7747, r9751;
}
{
add.f16x2 r9757, r8391, r8713;
}
{
mul.f16x2 r9760, r9757, r9650;
}
{
add.f16x2 r9763, r9754, r9760;
}
{
sub.f16x2 r9766, r8081, r9047;
}
{
mul.f16x2 r9769, r9766, r9648;
}
{
sub.f16x2 r9772, r8403, r8725;
}
{
mul.f16x2 r9775, r9772, r9651;
}
{
add.f16x2 r9778, r9769, r9775;
}
{
sub.f16x2 %20, r9763, r9778;
}
{
add.f16x2 r9784, r8069, r9035;
}
{
mul.f16x2 r9787, r9784, r9646;
}
{
add.f16x2 r9790, r7747, r9787;
}
{
add.f16x2 r9793, r8391, r8713;
}
{
mul.f16x2 r9796, r9793, r9650;
}
{
add.f16x2 r9799, r9790, r9796;
}
{
sub.f16x2 r9802, r8081, r9047;
}
{
mul.f16x2 r9805, r9802, r9648;
}
{
sub.f16x2 r9808, r8403, r8725;
}
{
mul.f16x2 r9811, r9808, r9651;
}
{
add.f16x2 r9814, r9805, r9811;
}
{
add.f16x2 %30, r9799, r9814;
}
{
add.f16x2 r9820, r8081, r9047;
}
{
mul.f16x2 r9823, r9820, r9642;
}
{
add.f16x2 r9826, r7759, r9823;
}
{
add.f16x2 r9829, r8403, r8725;
}
{
mul.f16x2 r9832, r9829, r9646;
}
{
add.f16x2 r9835, r9826, r9832;
}
{
sub.f16x2 r9838, r8069, r9035;
}
{
mul.f16x2 r9841, r9838, r9644;
}
{
sub.f16x2 r9844, r8391, r8713;
}
{
mul.f16x2 r9847, r9844, r9648;
}
{
add.f16x2 r9850, r9841, r9847;
}
{
add.f16x2 %11, r9835, r9850;
}
{
add.f16x2 r9856, r8081, r9047;
}
{
mul.f16x2 r9859, r9856, r9642;
}
{
add.f16x2 r9862, r7759, r9859;
}
{
add.f16x2 r9865, r8403, r8725;
}
{
mul.f16x2 r9868, r9865, r9646;
}
{
add.f16x2 r9871, r9862, r9868;
}
{
sub.f16x2 r9874, r8069, r9035;
}
{
mul.f16x2 r9877, r9874, r9644;
}
{
sub.f16x2 r9880, r8391, r8713;
}
{
mul.f16x2 r9883, r9880, r9648;
}
{
add.f16x2 r9886, r9877, r9883;
}
{
sub.f16x2 %41, r9871, r9886;
}
{
add.f16x2 r9892, r8081, r9047;
}
{
mul.f16x2 r9895, r9892, r9646;
}
{
add.f16x2 r9898, r7759, r9895;
}
{
add.f16x2 r9901, r8403, r8725;
}
{
mul.f16x2 r9904, r9901, r9650;
}
{
add.f16x2 r9907, r9898, r9904;
}
{
sub.f16x2 r9910, r8069, r9035;
}
{
mul.f16x2 r9913, r9910, r9648;
}
{
sub.f16x2 r9916, r8391, r8713;
}
{
mul.f16x2 r9919, r9916, r9651;
}
{
add.f16x2 r9922, r9913, r9919;
}
{
add.f16x2 %21, r9907, r9922;
}
{
add.f16x2 r9928, r8081, r9047;
}
{
mul.f16x2 r9931, r9928, r9646;
}
{
add.f16x2 r9934, r7759, r9931;
}
{
add.f16x2 r9937, r8403, r8725;
}
{
mul.f16x2 r9940, r9937, r9650;
}
{
add.f16x2 r9943, r9934, r9940;
}
{
sub.f16x2 r9946, r8069, r9035;
}
{
mul.f16x2 r9949, r9946, r9648;
}
{
sub.f16x2 r9952, r8391, r8713;
}
{
mul.f16x2 r9955, r9952, r9651;
}
{
add.f16x2 r9958, r9949, r9955;
}
{
sub.f16x2 %31, r9943, r9958;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9965, {low, high};
}
{
neg.f16x2 r9966, r9965;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r9968, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r9969, {low, high};
}
{
neg.f16x2 r9970, r9969;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r9972, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r9973, {low, high};
}
{
add.f16x2 r9974, r9392, r9440;
}
{
add.f16x2 r9977, r7795, r9974;
}
{
add.f16x2 r9980, r9408, r9424;
}
{
add.f16x2 %2, r9977, r9980;
}
{
add.f16x2 r9986, r9398, r9446;
}
{
add.f16x2 r9989, r7939, r9986;
}
{
add.f16x2 r9992, r9414, r9430;
}
{
add.f16x2 %3, r9989, r9992;
}
{
add.f16x2 r9998, r9392, r9440;
}
{
mul.f16x2 r10001, r9998, r9964;
}
{
add.f16x2 r10004, r7795, r10001;
}
{
add.f16x2 r10007, r9408, r9424;
}
{
mul.f16x2 r10010, r10007, r9968;
}
{
add.f16x2 r10013, r10004, r10010;
}
{
sub.f16x2 r10016, r9398, r9446;
}
{
mul.f16x2 r10019, r10016, r9966;
}
{
sub.f16x2 r10022, r9414, r9430;
}
{
mul.f16x2 r10025, r10022, r9970;
}
{
add.f16x2 r10028, r10019, r10025;
}
{
sub.f16x2 %12, r10013, r10028;
}
{
add.f16x2 r10034, r9392, r9440;
}
{
mul.f16x2 r10037, r10034, r9964;
}
{
add.f16x2 r10040, r7795, r10037;
}
{
add.f16x2 r10043, r9408, r9424;
}
{
mul.f16x2 r10046, r10043, r9968;
}
{
add.f16x2 r10049, r10040, r10046;
}
{
sub.f16x2 r10052, r9398, r9446;
}
{
mul.f16x2 r10055, r10052, r9966;
}
{
sub.f16x2 r10058, r9414, r9430;
}
{
mul.f16x2 r10061, r10058, r9970;
}
{
add.f16x2 r10064, r10055, r10061;
}
{
add.f16x2 %42, r10049, r10064;
}
{
add.f16x2 r10070, r9392, r9440;
}
{
mul.f16x2 r10073, r10070, r9968;
}
{
add.f16x2 r10076, r7795, r10073;
}
{
add.f16x2 r10079, r9408, r9424;
}
{
mul.f16x2 r10082, r10079, r9972;
}
{
add.f16x2 r10085, r10076, r10082;
}
{
sub.f16x2 r10088, r9398, r9446;
}
{
mul.f16x2 r10091, r10088, r9970;
}
{
sub.f16x2 r10094, r9414, r9430;
}
{
mul.f16x2 r10097, r10094, r9973;
}
{
add.f16x2 r10100, r10091, r10097;
}
{
sub.f16x2 %22, r10085, r10100;
}
{
add.f16x2 r10106, r9392, r9440;
}
{
mul.f16x2 r10109, r10106, r9968;
}
{
add.f16x2 r10112, r7795, r10109;
}
{
add.f16x2 r10115, r9408, r9424;
}
{
mul.f16x2 r10118, r10115, r9972;
}
{
add.f16x2 r10121, r10112, r10118;
}
{
sub.f16x2 r10124, r9398, r9446;
}
{
mul.f16x2 r10127, r10124, r9970;
}
{
sub.f16x2 r10130, r9414, r9430;
}
{
mul.f16x2 r10133, r10130, r9973;
}
{
add.f16x2 r10136, r10127, r10133;
}
{
add.f16x2 %32, r10121, r10136;
}
{
add.f16x2 r10142, r9398, r9446;
}
{
mul.f16x2 r10145, r10142, r9964;
}
{
add.f16x2 r10148, r7939, r10145;
}
{
add.f16x2 r10151, r9414, r9430;
}
{
mul.f16x2 r10154, r10151, r9968;
}
{
add.f16x2 r10157, r10148, r10154;
}
{
sub.f16x2 r10160, r9392, r9440;
}
{
mul.f16x2 r10163, r10160, r9966;
}
{
sub.f16x2 r10166, r9408, r9424;
}
{
mul.f16x2 r10169, r10166, r9970;
}
{
add.f16x2 r10172, r10163, r10169;
}
{
add.f16x2 %13, r10157, r10172;
}
{
add.f16x2 r10178, r9398, r9446;
}
{
mul.f16x2 r10181, r10178, r9964;
}
{
add.f16x2 r10184, r7939, r10181;
}
{
add.f16x2 r10187, r9414, r9430;
}
{
mul.f16x2 r10190, r10187, r9968;
}
{
add.f16x2 r10193, r10184, r10190;
}
{
sub.f16x2 r10196, r9392, r9440;
}
{
mul.f16x2 r10199, r10196, r9966;
}
{
sub.f16x2 r10202, r9408, r9424;
}
{
mul.f16x2 r10205, r10202, r9970;
}
{
add.f16x2 r10208, r10199, r10205;
}
{
sub.f16x2 %43, r10193, r10208;
}
{
add.f16x2 r10214, r9398, r9446;
}
{
mul.f16x2 r10217, r10214, r9968;
}
{
add.f16x2 r10220, r7939, r10217;
}
{
add.f16x2 r10223, r9414, r9430;
}
{
mul.f16x2 r10226, r10223, r9972;
}
{
add.f16x2 r10229, r10220, r10226;
}
{
sub.f16x2 r10232, r9392, r9440;
}
{
mul.f16x2 r10235, r10232, r9970;
}
{
sub.f16x2 r10238, r9408, r9424;
}
{
mul.f16x2 r10241, r10238, r9973;
}
{
add.f16x2 r10244, r10235, r10241;
}
{
add.f16x2 %23, r10229, r10244;
}
{
add.f16x2 r10250, r9398, r9446;
}
{
mul.f16x2 r10253, r10250, r9968;
}
{
add.f16x2 r10256, r7939, r10253;
}
{
add.f16x2 r10259, r9414, r9430;
}
{
mul.f16x2 r10262, r10259, r9972;
}
{
add.f16x2 r10265, r10256, r10262;
}
{
sub.f16x2 r10268, r9392, r9440;
}
{
mul.f16x2 r10271, r10268, r9970;
}
{
sub.f16x2 r10274, r9408, r9424;
}
{
mul.f16x2 r10277, r10274, r9973;
}
{
add.f16x2 r10280, r10271, r10277;
}
{
sub.f16x2 %33, r10265, r10280;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10286, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10287, {low, high};
}
{
neg.f16x2 r10288, r10287;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10290, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10291, {low, high};
}
{
neg.f16x2 r10292, r10291;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10294, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10295, {low, high};
}
{
add.f16x2 r10296, r9456, r9504;
}
{
add.f16x2 r10299, r7867, r10296;
}
{
add.f16x2 r10302, r9472, r9488;
}
{
add.f16x2 %4, r10299, r10302;
}
{
add.f16x2 r10308, r9462, r9510;
}
{
add.f16x2 r10311, r8011, r10308;
}
{
add.f16x2 r10314, r9478, r9494;
}
{
add.f16x2 %5, r10311, r10314;
}
{
add.f16x2 r10320, r9456, r9504;
}
{
mul.f16x2 r10323, r10320, r10286;
}
{
add.f16x2 r10326, r7867, r10323;
}
{
add.f16x2 r10329, r9472, r9488;
}
{
mul.f16x2 r10332, r10329, r10290;
}
{
add.f16x2 r10335, r10326, r10332;
}
{
sub.f16x2 r10338, r9462, r9510;
}
{
mul.f16x2 r10341, r10338, r10288;
}
{
sub.f16x2 r10344, r9478, r9494;
}
{
mul.f16x2 r10347, r10344, r10292;
}
{
add.f16x2 r10350, r10341, r10347;
}
{
sub.f16x2 %14, r10335, r10350;
}
{
add.f16x2 r10356, r9456, r9504;
}
{
mul.f16x2 r10359, r10356, r10286;
}
{
add.f16x2 r10362, r7867, r10359;
}
{
add.f16x2 r10365, r9472, r9488;
}
{
mul.f16x2 r10368, r10365, r10290;
}
{
add.f16x2 r10371, r10362, r10368;
}
{
sub.f16x2 r10374, r9462, r9510;
}
{
mul.f16x2 r10377, r10374, r10288;
}
{
sub.f16x2 r10380, r9478, r9494;
}
{
mul.f16x2 r10383, r10380, r10292;
}
{
add.f16x2 r10386, r10377, r10383;
}
{
add.f16x2 %44, r10371, r10386;
}
{
add.f16x2 r10392, r9456, r9504;
}
{
mul.f16x2 r10395, r10392, r10290;
}
{
add.f16x2 r10398, r7867, r10395;
}
{
add.f16x2 r10401, r9472, r9488;
}
{
mul.f16x2 r10404, r10401, r10294;
}
{
add.f16x2 r10407, r10398, r10404;
}
{
sub.f16x2 r10410, r9462, r9510;
}
{
mul.f16x2 r10413, r10410, r10292;
}
{
sub.f16x2 r10416, r9478, r9494;
}
{
mul.f16x2 r10419, r10416, r10295;
}
{
add.f16x2 r10422, r10413, r10419;
}
{
sub.f16x2 %24, r10407, r10422;
}
{
add.f16x2 r10428, r9456, r9504;
}
{
mul.f16x2 r10431, r10428, r10290;
}
{
add.f16x2 r10434, r7867, r10431;
}
{
add.f16x2 r10437, r9472, r9488;
}
{
mul.f16x2 r10440, r10437, r10294;
}
{
add.f16x2 r10443, r10434, r10440;
}
{
sub.f16x2 r10446, r9462, r9510;
}
{
mul.f16x2 r10449, r10446, r10292;
}
{
sub.f16x2 r10452, r9478, r9494;
}
{
mul.f16x2 r10455, r10452, r10295;
}
{
add.f16x2 r10458, r10449, r10455;
}
{
add.f16x2 %34, r10443, r10458;
}
{
add.f16x2 r10464, r9462, r9510;
}
{
mul.f16x2 r10467, r10464, r10286;
}
{
add.f16x2 r10470, r8011, r10467;
}
{
add.f16x2 r10473, r9478, r9494;
}
{
mul.f16x2 r10476, r10473, r10290;
}
{
add.f16x2 r10479, r10470, r10476;
}
{
sub.f16x2 r10482, r9456, r9504;
}
{
mul.f16x2 r10485, r10482, r10288;
}
{
sub.f16x2 r10488, r9472, r9488;
}
{
mul.f16x2 r10491, r10488, r10292;
}
{
add.f16x2 r10494, r10485, r10491;
}
{
add.f16x2 %15, r10479, r10494;
}
{
add.f16x2 r10500, r9462, r9510;
}
{
mul.f16x2 r10503, r10500, r10286;
}
{
add.f16x2 r10506, r8011, r10503;
}
{
add.f16x2 r10509, r9478, r9494;
}
{
mul.f16x2 r10512, r10509, r10290;
}
{
add.f16x2 r10515, r10506, r10512;
}
{
sub.f16x2 r10518, r9456, r9504;
}
{
mul.f16x2 r10521, r10518, r10288;
}
{
sub.f16x2 r10524, r9472, r9488;
}
{
mul.f16x2 r10527, r10524, r10292;
}
{
add.f16x2 r10530, r10521, r10527;
}
{
sub.f16x2 %45, r10515, r10530;
}
{
add.f16x2 r10536, r9462, r9510;
}
{
mul.f16x2 r10539, r10536, r10290;
}
{
add.f16x2 r10542, r8011, r10539;
}
{
add.f16x2 r10545, r9478, r9494;
}
{
mul.f16x2 r10548, r10545, r10294;
}
{
add.f16x2 r10551, r10542, r10548;
}
{
sub.f16x2 r10554, r9456, r9504;
}
{
mul.f16x2 r10557, r10554, r10292;
}
{
sub.f16x2 r10560, r9472, r9488;
}
{
mul.f16x2 r10563, r10560, r10295;
}
{
add.f16x2 r10566, r10557, r10563;
}
{
add.f16x2 %25, r10551, r10566;
}
{
add.f16x2 r10572, r9462, r9510;
}
{
mul.f16x2 r10575, r10572, r10290;
}
{
add.f16x2 r10578, r8011, r10575;
}
{
add.f16x2 r10581, r9478, r9494;
}
{
mul.f16x2 r10584, r10581, r10294;
}
{
add.f16x2 r10587, r10578, r10584;
}
{
sub.f16x2 r10590, r9456, r9504;
}
{
mul.f16x2 r10593, r10590, r10292;
}
{
sub.f16x2 r10596, r9472, r9488;
}
{
mul.f16x2 r10599, r10596, r10295;
}
{
add.f16x2 r10602, r10593, r10599;
}
{
sub.f16x2 %35, r10587, r10602;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10608, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10609, {low, high};
}
{
neg.f16x2 r10610, r10609;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10612, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10613, {low, high};
}
{
neg.f16x2 r10614, r10613;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10616, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10617, {low, high};
}
{
add.f16x2 r10618, r9520, r9568;
}
{
add.f16x2 r10621, r7903, r10618;
}
{
add.f16x2 r10624, r9536, r9552;
}
{
add.f16x2 %6, r10621, r10624;
}
{
add.f16x2 r10630, r9526, r9574;
}
{
add.f16x2 r10633, r8047, r10630;
}
{
add.f16x2 r10636, r9542, r9558;
}
{
add.f16x2 %7, r10633, r10636;
}
{
add.f16x2 r10642, r9520, r9568;
}
{
mul.f16x2 r10645, r10642, r10608;
}
{
add.f16x2 r10648, r7903, r10645;
}
{
add.f16x2 r10651, r9536, r9552;
}
{
mul.f16x2 r10654, r10651, r10612;
}
{
add.f16x2 r10657, r10648, r10654;
}
{
sub.f16x2 r10660, r9526, r9574;
}
{
mul.f16x2 r10663, r10660, r10610;
}
{
sub.f16x2 r10666, r9542, r9558;
}
{
mul.f16x2 r10669, r10666, r10614;
}
{
add.f16x2 r10672, r10663, r10669;
}
{
sub.f16x2 %16, r10657, r10672;
}
{
add.f16x2 r10678, r9520, r9568;
}
{
mul.f16x2 r10681, r10678, r10608;
}
{
add.f16x2 r10684, r7903, r10681;
}
{
add.f16x2 r10687, r9536, r9552;
}
{
mul.f16x2 r10690, r10687, r10612;
}
{
add.f16x2 r10693, r10684, r10690;
}
{
sub.f16x2 r10696, r9526, r9574;
}
{
mul.f16x2 r10699, r10696, r10610;
}
{
sub.f16x2 r10702, r9542, r9558;
}
{
mul.f16x2 r10705, r10702, r10614;
}
{
add.f16x2 r10708, r10699, r10705;
}
{
add.f16x2 %46, r10693, r10708;
}
{
add.f16x2 r10714, r9520, r9568;
}
{
mul.f16x2 r10717, r10714, r10612;
}
{
add.f16x2 r10720, r7903, r10717;
}
{
add.f16x2 r10723, r9536, r9552;
}
{
mul.f16x2 r10726, r10723, r10616;
}
{
add.f16x2 r10729, r10720, r10726;
}
{
sub.f16x2 r10732, r9526, r9574;
}
{
mul.f16x2 r10735, r10732, r10614;
}
{
sub.f16x2 r10738, r9542, r9558;
}
{
mul.f16x2 r10741, r10738, r10617;
}
{
add.f16x2 r10744, r10735, r10741;
}
{
sub.f16x2 %26, r10729, r10744;
}
{
add.f16x2 r10750, r9520, r9568;
}
{
mul.f16x2 r10753, r10750, r10612;
}
{
add.f16x2 r10756, r7903, r10753;
}
{
add.f16x2 r10759, r9536, r9552;
}
{
mul.f16x2 r10762, r10759, r10616;
}
{
add.f16x2 r10765, r10756, r10762;
}
{
sub.f16x2 r10768, r9526, r9574;
}
{
mul.f16x2 r10771, r10768, r10614;
}
{
sub.f16x2 r10774, r9542, r9558;
}
{
mul.f16x2 r10777, r10774, r10617;
}
{
add.f16x2 r10780, r10771, r10777;
}
{
add.f16x2 %36, r10765, r10780;
}
{
add.f16x2 r10786, r9526, r9574;
}
{
mul.f16x2 r10789, r10786, r10608;
}
{
add.f16x2 r10792, r8047, r10789;
}
{
add.f16x2 r10795, r9542, r9558;
}
{
mul.f16x2 r10798, r10795, r10612;
}
{
add.f16x2 r10801, r10792, r10798;
}
{
sub.f16x2 r10804, r9520, r9568;
}
{
mul.f16x2 r10807, r10804, r10610;
}
{
sub.f16x2 r10810, r9536, r9552;
}
{
mul.f16x2 r10813, r10810, r10614;
}
{
add.f16x2 r10816, r10807, r10813;
}
{
add.f16x2 %17, r10801, r10816;
}
{
add.f16x2 r10822, r9526, r9574;
}
{
mul.f16x2 r10825, r10822, r10608;
}
{
add.f16x2 r10828, r8047, r10825;
}
{
add.f16x2 r10831, r9542, r9558;
}
{
mul.f16x2 r10834, r10831, r10612;
}
{
add.f16x2 r10837, r10828, r10834;
}
{
sub.f16x2 r10840, r9520, r9568;
}
{
mul.f16x2 r10843, r10840, r10610;
}
{
sub.f16x2 r10846, r9536, r9552;
}
{
mul.f16x2 r10849, r10846, r10614;
}
{
add.f16x2 r10852, r10843, r10849;
}
{
sub.f16x2 %47, r10837, r10852;
}
{
add.f16x2 r10858, r9526, r9574;
}
{
mul.f16x2 r10861, r10858, r10612;
}
{
add.f16x2 r10864, r8047, r10861;
}
{
add.f16x2 r10867, r9542, r9558;
}
{
mul.f16x2 r10870, r10867, r10616;
}
{
add.f16x2 r10873, r10864, r10870;
}
{
sub.f16x2 r10876, r9520, r9568;
}
{
mul.f16x2 r10879, r10876, r10614;
}
{
sub.f16x2 r10882, r9536, r9552;
}
{
mul.f16x2 r10885, r10882, r10617;
}
{
add.f16x2 r10888, r10879, r10885;
}
{
add.f16x2 %27, r10873, r10888;
}
{
add.f16x2 r10894, r9526, r9574;
}
{
mul.f16x2 r10897, r10894, r10612;
}
{
add.f16x2 r10900, r8047, r10897;
}
{
add.f16x2 r10903, r9542, r9558;
}
{
mul.f16x2 r10906, r10903, r10616;
}
{
add.f16x2 r10909, r10900, r10906;
}
{
sub.f16x2 r10912, r9520, r9568;
}
{
mul.f16x2 r10915, r10912, r10614;
}
{
sub.f16x2 r10918, r9536, r9552;
}
{
mul.f16x2 r10921, r10918, r10617;
}
{
add.f16x2 r10924, r10915, r10921;
}
{
sub.f16x2 %37, r10909, r10924;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10930, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10931, {low, high};
}
{
neg.f16x2 r10932, r10931;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f210;
cvt.rn.f16.f32 high, f210;
mov.b32 r10934, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f212;
cvt.rn.f16.f32 high, f212;
mov.b32 r10935, {low, high};
}
{
neg.f16x2 r10936, r10935;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f214;
cvt.rn.f16.f32 high, f214;
mov.b32 r10938, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f216;
cvt.rn.f16.f32 high, f216;
mov.b32 r10939, {low, high};
}
{
add.f16x2 r10940, r9584, r9632;
}
{
add.f16x2 r10943, r7831, r10940;
}
{
add.f16x2 r10946, r9600, r9616;
}
{
add.f16x2 %8, r10943, r10946;
}
{
add.f16x2 r10952, r9590, r9638;
}
{
add.f16x2 r10955, r7975, r10952;
}
{
add.f16x2 r10958, r9606, r9622;
}
{
add.f16x2 %9, r10955, r10958;
}
{
add.f16x2 r10964, r9584, r9632;
}
{
mul.f16x2 r10967, r10964, r10930;
}
{
add.f16x2 r10970, r7831, r10967;
}
{
add.f16x2 r10973, r9600, r9616;
}
{
mul.f16x2 r10976, r10973, r10934;
}
{
add.f16x2 r10979, r10970, r10976;
}
{
sub.f16x2 r10982, r9590, r9638;
}
{
mul.f16x2 r10985, r10982, r10932;
}
{
sub.f16x2 r10988, r9606, r9622;
}
{
mul.f16x2 r10991, r10988, r10936;
}
{
add.f16x2 r10994, r10985, r10991;
}
{
sub.f16x2 %18, r10979, r10994;
}
{
add.f16x2 r11000, r9584, r9632;
}
{
mul.f16x2 r11003, r11000, r10930;
}
{
add.f16x2 r11006, r7831, r11003;
}
{
add.f16x2 r11009, r9600, r9616;
}
{
mul.f16x2 r11012, r11009, r10934;
}
{
add.f16x2 r11015, r11006, r11012;
}
{
sub.f16x2 r11018, r9590, r9638;
}
{
mul.f16x2 r11021, r11018, r10932;
}
{
sub.f16x2 r11024, r9606, r9622;
}
{
mul.f16x2 r11027, r11024, r10936;
}
{
add.f16x2 r11030, r11021, r11027;
}
{
add.f16x2 %48, r11015, r11030;
}
{
add.f16x2 r11036, r9584, r9632;
}
{
mul.f16x2 r11039, r11036, r10934;
}
{
add.f16x2 r11042, r7831, r11039;
}
{
add.f16x2 r11045, r9600, r9616;
}
{
mul.f16x2 r11048, r11045, r10938;
}
{
add.f16x2 r11051, r11042, r11048;
}
{
sub.f16x2 r11054, r9590, r9638;
}
{
mul.f16x2 r11057, r11054, r10936;
}
{
sub.f16x2 r11060, r9606, r9622;
}
{
mul.f16x2 r11063, r11060, r10939;
}
{
add.f16x2 r11066, r11057, r11063;
}
{
sub.f16x2 %28, r11051, r11066;
}
{
add.f16x2 r11072, r9584, r9632;
}
{
mul.f16x2 r11075, r11072, r10934;
}
{
add.f16x2 r11078, r7831, r11075;
}
{
add.f16x2 r11081, r9600, r9616;
}
{
mul.f16x2 r11084, r11081, r10938;
}
{
add.f16x2 r11087, r11078, r11084;
}
{
sub.f16x2 r11090, r9590, r9638;
}
{
mul.f16x2 r11093, r11090, r10936;
}
{
sub.f16x2 r11096, r9606, r9622;
}
{
mul.f16x2 r11099, r11096, r10939;
}
{
add.f16x2 r11102, r11093, r11099;
}
{
add.f16x2 %38, r11087, r11102;
}
{
add.f16x2 r11108, r9590, r9638;
}
{
mul.f16x2 r11111, r11108, r10930;
}
{
add.f16x2 r11114, r7975, r11111;
}
{
add.f16x2 r11117, r9606, r9622;
}
{
mul.f16x2 r11120, r11117, r10934;
}
{
add.f16x2 r11123, r11114, r11120;
}
{
sub.f16x2 r11126, r9584, r9632;
}
{
mul.f16x2 r11129, r11126, r10932;
}
{
sub.f16x2 r11132, r9600, r9616;
}
{
mul.f16x2 r11135, r11132, r10936;
}
{
add.f16x2 r11138, r11129, r11135;
}
{
add.f16x2 %19, r11123, r11138;
}
{
add.f16x2 r11144, r9590, r9638;
}
{
mul.f16x2 r11147, r11144, r10930;
}
{
add.f16x2 r11150, r7975, r11147;
}
{
add.f16x2 r11153, r9606, r9622;
}
{
mul.f16x2 r11156, r11153, r10934;
}
{
add.f16x2 r11159, r11150, r11156;
}
{
sub.f16x2 r11162, r9584, r9632;
}
{
mul.f16x2 r11165, r11162, r10932;
}
{
sub.f16x2 r11168, r9600, r9616;
}
{
mul.f16x2 r11171, r11168, r10936;
}
{
add.f16x2 r11174, r11165, r11171;
}
{
sub.f16x2 %49, r11159, r11174;
}
{
add.f16x2 r11180, r9590, r9638;
}
{
mul.f16x2 r11183, r11180, r10934;
}
{
add.f16x2 r11186, r7975, r11183;
}
{
add.f16x2 r11189, r9606, r9622;
}
{
mul.f16x2 r11192, r11189, r10938;
}
{
add.f16x2 r11195, r11186, r11192;
}
{
sub.f16x2 r11198, r9584, r9632;
}
{
mul.f16x2 r11201, r11198, r10936;
}
{
sub.f16x2 r11204, r9600, r9616;
}
{
mul.f16x2 r11207, r11204, r10939;
}
{
add.f16x2 r11210, r11201, r11207;
}
{
add.f16x2 %29, r11195, r11210;
}
{
add.f16x2 r11216, r9590, r9638;
}
{
mul.f16x2 r11219, r11216, r10934;
}
{
add.f16x2 r11222, r7975, r11219;
}
{
add.f16x2 r11225, r9606, r9622;
}
{
mul.f16x2 r11228, r11225, r10938;
}
{
add.f16x2 r11231, r11222, r11228;
}
{
sub.f16x2 r11234, r9584, r9632;
}
{
mul.f16x2 r11237, r11234, r10936;
}
{
sub.f16x2 r11240, r9600, r9616;
}
{
mul.f16x2 r11243, r11240, r10939;
}
{
add.f16x2 r11246, r11237, r11243;
}
{
sub.f16x2 %39, r11231, r11246;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)));
};


#endif
