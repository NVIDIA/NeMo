#ifndef CUFFTDX_FFT_3125_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_3125_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<913, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<603>;
.reg .b32 r<10458>;
.reg .b64 rd<6>;
mov.u32 r10392, %50;
mov.u32 r10457, %tid.y;
mad.lo.s32 r10393, r10457, 25000, r10392;
mov.u32 r10394, %tid.x;
mov.f32 f594, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1, {low, high};
}
mov.f32 f596, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2, {low, high};
}
mov.f32 f590, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3, {low, high};
}
mov.f32 f592, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %95, %81;
}
{
add.f16x2 r12, %54, r9;
}
{
add.f16x2 r15, %59, %94;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %70, %58;
}
{
add.f16x2 r24, %79, r21;
}
{
add.f16x2 r27, %87, %69;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %95, %81;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %54, r36;
}
{
add.f16x2 r42, %59, %94;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %70, %58;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %87, %69;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %95, %81;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %54, r72;
}
{
add.f16x2 r78, %59, %94;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %70, %58;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %87, %69;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %95, %81;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %54, r108;
}
{
add.f16x2 r114, %59, %94;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %70, %58;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %87, %69;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %95, %81;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %54, r144;
}
{
add.f16x2 r150, %59, %94;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %70, %58;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %87, %69;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %70, %58;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %79, r180;
}
{
add.f16x2 r186, %87, %69;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %95, %81;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %59, %94;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %70, %58;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %79, r216;
}
{
add.f16x2 r222, %87, %69;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %95, %81;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %59, %94;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %70, %58;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %79, r252;
}
{
add.f16x2 r258, %87, %69;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %95, %81;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %59, %94;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %70, %58;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %79, r288;
}
{
add.f16x2 r294, %87, %69;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %95, %81;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %59, %94;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r326, {low, high};
}
{
neg.f16x2 r327, r326;
}
{
add.f16x2 r329, %97, %83;
}
{
add.f16x2 r332, %55, r329;
}
{
add.f16x2 r335, %62, %96;
}
{
add.f16x2 r338, r332, r335;
}
{
add.f16x2 r341, %72, %61;
}
{
add.f16x2 r344, %80, r341;
}
{
add.f16x2 r347, %89, %71;
}
{
add.f16x2 r350, r344, r347;
}
{
add.f16x2 r353, %97, %83;
}
{
mul.f16x2 r356, r353, r321;
}
{
add.f16x2 r359, %55, r356;
}
{
add.f16x2 r362, %62, %96;
}
{
mul.f16x2 r365, r362, r323;
}
{
add.f16x2 r368, r359, r365;
}
{
sub.f16x2 r371, %72, %61;
}
{
mul.f16x2 r374, r371, r322;
}
{
sub.f16x2 r377, %89, %71;
}
{
mul.f16x2 r380, r377, r324;
}
{
add.f16x2 r383, r374, r380;
}
{
sub.f16x2 r386, r368, r383;
}
{
add.f16x2 r389, %97, %83;
}
{
mul.f16x2 r392, r389, r321;
}
{
add.f16x2 r395, %55, r392;
}
{
add.f16x2 r398, %62, %96;
}
{
mul.f16x2 r401, r398, r323;
}
{
add.f16x2 r404, r395, r401;
}
{
sub.f16x2 r407, %72, %61;
}
{
mul.f16x2 r410, r407, r322;
}
{
sub.f16x2 r413, %89, %71;
}
{
mul.f16x2 r416, r413, r324;
}
{
add.f16x2 r419, r410, r416;
}
{
add.f16x2 r422, r404, r419;
}
{
add.f16x2 r425, %97, %83;
}
{
mul.f16x2 r428, r425, r323;
}
{
add.f16x2 r431, %55, r428;
}
{
add.f16x2 r434, %62, %96;
}
{
mul.f16x2 r437, r434, r325;
}
{
add.f16x2 r440, r431, r437;
}
{
sub.f16x2 r443, %72, %61;
}
{
mul.f16x2 r446, r443, r324;
}
{
sub.f16x2 r449, %89, %71;
}
{
mul.f16x2 r452, r449, r327;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, r440, r455;
}
{
add.f16x2 r461, %97, %83;
}
{
mul.f16x2 r464, r461, r323;
}
{
add.f16x2 r467, %55, r464;
}
{
add.f16x2 r470, %62, %96;
}
{
mul.f16x2 r473, r470, r325;
}
{
add.f16x2 r476, r467, r473;
}
{
sub.f16x2 r479, %72, %61;
}
{
mul.f16x2 r482, r479, r324;
}
{
sub.f16x2 r485, %89, %71;
}
{
mul.f16x2 r488, r485, r327;
}
{
add.f16x2 r491, r482, r488;
}
{
add.f16x2 r494, r476, r491;
}
{
add.f16x2 r497, %72, %61;
}
{
mul.f16x2 r500, r497, r321;
}
{
add.f16x2 r503, %80, r500;
}
{
add.f16x2 r506, %89, %71;
}
{
mul.f16x2 r509, r506, r323;
}
{
add.f16x2 r512, r503, r509;
}
{
sub.f16x2 r515, %97, %83;
}
{
mul.f16x2 r518, r515, r322;
}
{
sub.f16x2 r521, %62, %96;
}
{
mul.f16x2 r524, r521, r324;
}
{
add.f16x2 r527, r518, r524;
}
{
add.f16x2 r530, r512, r527;
}
{
add.f16x2 r533, %72, %61;
}
{
mul.f16x2 r536, r533, r321;
}
{
add.f16x2 r539, %80, r536;
}
{
add.f16x2 r542, %89, %71;
}
{
mul.f16x2 r545, r542, r323;
}
{
add.f16x2 r548, r539, r545;
}
{
sub.f16x2 r551, %97, %83;
}
{
mul.f16x2 r554, r551, r322;
}
{
sub.f16x2 r557, %62, %96;
}
{
mul.f16x2 r560, r557, r324;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, r548, r563;
}
{
add.f16x2 r569, %72, %61;
}
{
mul.f16x2 r572, r569, r323;
}
{
add.f16x2 r575, %80, r572;
}
{
add.f16x2 r578, %89, %71;
}
{
mul.f16x2 r581, r578, r325;
}
{
add.f16x2 r584, r575, r581;
}
{
sub.f16x2 r587, %97, %83;
}
{
mul.f16x2 r590, r587, r324;
}
{
sub.f16x2 r593, %62, %96;
}
{
mul.f16x2 r596, r593, r327;
}
{
add.f16x2 r599, r590, r596;
}
{
add.f16x2 r602, r584, r599;
}
{
add.f16x2 r605, %72, %61;
}
{
mul.f16x2 r608, r605, r323;
}
{
add.f16x2 r611, %80, r608;
}
{
add.f16x2 r614, %89, %71;
}
{
mul.f16x2 r617, r614, r325;
}
{
add.f16x2 r620, r611, r617;
}
{
sub.f16x2 r623, %97, %83;
}
{
mul.f16x2 r626, r623, r324;
}
{
sub.f16x2 r629, %62, %96;
}
{
mul.f16x2 r632, r629, r327;
}
{
add.f16x2 r635, r626, r632;
}
{
sub.f16x2 r638, r620, r635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r644, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
add.f16x2 r649, %99, %85;
}
{
add.f16x2 r652, %56, r649;
}
{
add.f16x2 r655, %64, %98;
}
{
add.f16x2 r658, r652, r655;
}
{
add.f16x2 r661, %74, %63;
}
{
add.f16x2 r664, %82, r661;
}
{
add.f16x2 r667, %91, %73;
}
{
add.f16x2 r670, r664, r667;
}
{
add.f16x2 r673, %99, %85;
}
{
mul.f16x2 r676, r673, r641;
}
{
add.f16x2 r679, %56, r676;
}
{
add.f16x2 r682, %64, %98;
}
{
mul.f16x2 r685, r682, r643;
}
{
add.f16x2 r688, r679, r685;
}
{
sub.f16x2 r691, %74, %63;
}
{
mul.f16x2 r694, r691, r642;
}
{
sub.f16x2 r697, %91, %73;
}
{
mul.f16x2 r700, r697, r644;
}
{
add.f16x2 r703, r694, r700;
}
{
sub.f16x2 r706, r688, r703;
}
{
add.f16x2 r709, %99, %85;
}
{
mul.f16x2 r712, r709, r641;
}
{
add.f16x2 r715, %56, r712;
}
{
add.f16x2 r718, %64, %98;
}
{
mul.f16x2 r721, r718, r643;
}
{
add.f16x2 r724, r715, r721;
}
{
sub.f16x2 r727, %74, %63;
}
{
mul.f16x2 r730, r727, r642;
}
{
sub.f16x2 r733, %91, %73;
}
{
mul.f16x2 r736, r733, r644;
}
{
add.f16x2 r739, r730, r736;
}
{
add.f16x2 r742, r724, r739;
}
{
add.f16x2 r745, %99, %85;
}
{
mul.f16x2 r748, r745, r643;
}
{
add.f16x2 r751, %56, r748;
}
{
add.f16x2 r754, %64, %98;
}
{
mul.f16x2 r757, r754, r645;
}
{
add.f16x2 r760, r751, r757;
}
{
sub.f16x2 r763, %74, %63;
}
{
mul.f16x2 r766, r763, r644;
}
{
sub.f16x2 r769, %91, %73;
}
{
mul.f16x2 r772, r769, r647;
}
{
add.f16x2 r775, r766, r772;
}
{
sub.f16x2 r778, r760, r775;
}
{
add.f16x2 r781, %99, %85;
}
{
mul.f16x2 r784, r781, r643;
}
{
add.f16x2 r787, %56, r784;
}
{
add.f16x2 r790, %64, %98;
}
{
mul.f16x2 r793, r790, r645;
}
{
add.f16x2 r796, r787, r793;
}
{
sub.f16x2 r799, %74, %63;
}
{
mul.f16x2 r802, r799, r644;
}
{
sub.f16x2 r805, %91, %73;
}
{
mul.f16x2 r808, r805, r647;
}
{
add.f16x2 r811, r802, r808;
}
{
add.f16x2 r814, r796, r811;
}
{
add.f16x2 r817, %74, %63;
}
{
mul.f16x2 r820, r817, r641;
}
{
add.f16x2 r823, %82, r820;
}
{
add.f16x2 r826, %91, %73;
}
{
mul.f16x2 r829, r826, r643;
}
{
add.f16x2 r832, r823, r829;
}
{
sub.f16x2 r835, %99, %85;
}
{
mul.f16x2 r838, r835, r642;
}
{
sub.f16x2 r841, %64, %98;
}
{
mul.f16x2 r844, r841, r644;
}
{
add.f16x2 r847, r838, r844;
}
{
add.f16x2 r850, r832, r847;
}
{
add.f16x2 r853, %74, %63;
}
{
mul.f16x2 r856, r853, r641;
}
{
add.f16x2 r859, %82, r856;
}
{
add.f16x2 r862, %91, %73;
}
{
mul.f16x2 r865, r862, r643;
}
{
add.f16x2 r868, r859, r865;
}
{
sub.f16x2 r871, %99, %85;
}
{
mul.f16x2 r874, r871, r642;
}
{
sub.f16x2 r877, %64, %98;
}
{
mul.f16x2 r880, r877, r644;
}
{
add.f16x2 r883, r874, r880;
}
{
sub.f16x2 r886, r868, r883;
}
{
add.f16x2 r889, %74, %63;
}
{
mul.f16x2 r892, r889, r643;
}
{
add.f16x2 r895, %82, r892;
}
{
add.f16x2 r898, %91, %73;
}
{
mul.f16x2 r901, r898, r645;
}
{
add.f16x2 r904, r895, r901;
}
{
sub.f16x2 r907, %99, %85;
}
{
mul.f16x2 r910, r907, r644;
}
{
sub.f16x2 r913, %64, %98;
}
{
mul.f16x2 r916, r913, r647;
}
{
add.f16x2 r919, r910, r916;
}
{
add.f16x2 r922, r904, r919;
}
{
add.f16x2 r925, %74, %63;
}
{
mul.f16x2 r928, r925, r643;
}
{
add.f16x2 r931, %82, r928;
}
{
add.f16x2 r934, %91, %73;
}
{
mul.f16x2 r937, r934, r645;
}
{
add.f16x2 r940, r931, r937;
}
{
sub.f16x2 r943, %99, %85;
}
{
mul.f16x2 r946, r943, r644;
}
{
sub.f16x2 r949, %64, %98;
}
{
mul.f16x2 r952, r949, r647;
}
{
add.f16x2 r955, r946, r952;
}
{
sub.f16x2 r958, r940, r955;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r961, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r962, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r965, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r966, {low, high};
}
{
neg.f16x2 r967, r966;
}
{
add.f16x2 r969, %51, %88;
}
{
add.f16x2 r972, %57, r969;
}
{
add.f16x2 r975, %66, %100;
}
{
add.f16x2 r978, r972, r975;
}
{
add.f16x2 r981, %76, %65;
}
{
add.f16x2 r984, %84, r981;
}
{
add.f16x2 r987, %92, %75;
}
{
add.f16x2 r990, r984, r987;
}
{
add.f16x2 r993, %51, %88;
}
{
mul.f16x2 r996, r993, r961;
}
{
add.f16x2 r999, %57, r996;
}
{
add.f16x2 r1002, %66, %100;
}
{
mul.f16x2 r1005, r1002, r963;
}
{
add.f16x2 r1008, r999, r1005;
}
{
sub.f16x2 r1011, %76, %65;
}
{
mul.f16x2 r1014, r1011, r962;
}
{
sub.f16x2 r1017, %92, %75;
}
{
mul.f16x2 r1020, r1017, r964;
}
{
add.f16x2 r1023, r1014, r1020;
}
{
sub.f16x2 r1026, r1008, r1023;
}
{
add.f16x2 r1029, %51, %88;
}
{
mul.f16x2 r1032, r1029, r961;
}
{
add.f16x2 r1035, %57, r1032;
}
{
add.f16x2 r1038, %66, %100;
}
{
mul.f16x2 r1041, r1038, r963;
}
{
add.f16x2 r1044, r1035, r1041;
}
{
sub.f16x2 r1047, %76, %65;
}
{
mul.f16x2 r1050, r1047, r962;
}
{
sub.f16x2 r1053, %92, %75;
}
{
mul.f16x2 r1056, r1053, r964;
}
{
add.f16x2 r1059, r1050, r1056;
}
{
add.f16x2 r1062, r1044, r1059;
}
{
add.f16x2 r1065, %51, %88;
}
{
mul.f16x2 r1068, r1065, r963;
}
{
add.f16x2 r1071, %57, r1068;
}
{
add.f16x2 r1074, %66, %100;
}
{
mul.f16x2 r1077, r1074, r965;
}
{
add.f16x2 r1080, r1071, r1077;
}
{
sub.f16x2 r1083, %76, %65;
}
{
mul.f16x2 r1086, r1083, r964;
}
{
sub.f16x2 r1089, %92, %75;
}
{
mul.f16x2 r1092, r1089, r967;
}
{
add.f16x2 r1095, r1086, r1092;
}
{
sub.f16x2 r1098, r1080, r1095;
}
{
add.f16x2 r1101, %51, %88;
}
{
mul.f16x2 r1104, r1101, r963;
}
{
add.f16x2 r1107, %57, r1104;
}
{
add.f16x2 r1110, %66, %100;
}
{
mul.f16x2 r1113, r1110, r965;
}
{
add.f16x2 r1116, r1107, r1113;
}
{
sub.f16x2 r1119, %76, %65;
}
{
mul.f16x2 r1122, r1119, r964;
}
{
sub.f16x2 r1125, %92, %75;
}
{
mul.f16x2 r1128, r1125, r967;
}
{
add.f16x2 r1131, r1122, r1128;
}
{
add.f16x2 r1134, r1116, r1131;
}
{
add.f16x2 r1137, %76, %65;
}
{
mul.f16x2 r1140, r1137, r961;
}
{
add.f16x2 r1143, %84, r1140;
}
{
add.f16x2 r1146, %92, %75;
}
{
mul.f16x2 r1149, r1146, r963;
}
{
add.f16x2 r1152, r1143, r1149;
}
{
sub.f16x2 r1155, %51, %88;
}
{
mul.f16x2 r1158, r1155, r962;
}
{
sub.f16x2 r1161, %66, %100;
}
{
mul.f16x2 r1164, r1161, r964;
}
{
add.f16x2 r1167, r1158, r1164;
}
{
add.f16x2 r1170, r1152, r1167;
}
{
add.f16x2 r1173, %76, %65;
}
{
mul.f16x2 r1176, r1173, r961;
}
{
add.f16x2 r1179, %84, r1176;
}
{
add.f16x2 r1182, %92, %75;
}
{
mul.f16x2 r1185, r1182, r963;
}
{
add.f16x2 r1188, r1179, r1185;
}
{
sub.f16x2 r1191, %51, %88;
}
{
mul.f16x2 r1194, r1191, r962;
}
{
sub.f16x2 r1197, %66, %100;
}
{
mul.f16x2 r1200, r1197, r964;
}
{
add.f16x2 r1203, r1194, r1200;
}
{
sub.f16x2 r1206, r1188, r1203;
}
{
add.f16x2 r1209, %76, %65;
}
{
mul.f16x2 r1212, r1209, r963;
}
{
add.f16x2 r1215, %84, r1212;
}
{
add.f16x2 r1218, %92, %75;
}
{
mul.f16x2 r1221, r1218, r965;
}
{
add.f16x2 r1224, r1215, r1221;
}
{
sub.f16x2 r1227, %51, %88;
}
{
mul.f16x2 r1230, r1227, r964;
}
{
sub.f16x2 r1233, %66, %100;
}
{
mul.f16x2 r1236, r1233, r967;
}
{
add.f16x2 r1239, r1230, r1236;
}
{
add.f16x2 r1242, r1224, r1239;
}
{
add.f16x2 r1245, %76, %65;
}
{
mul.f16x2 r1248, r1245, r963;
}
{
add.f16x2 r1251, %84, r1248;
}
{
add.f16x2 r1254, %92, %75;
}
{
mul.f16x2 r1257, r1254, r965;
}
{
add.f16x2 r1260, r1251, r1257;
}
{
sub.f16x2 r1263, %51, %88;
}
{
mul.f16x2 r1266, r1263, r964;
}
{
sub.f16x2 r1269, %66, %100;
}
{
mul.f16x2 r1272, r1269, r967;
}
{
add.f16x2 r1275, r1266, r1272;
}
{
sub.f16x2 r1278, r1260, r1275;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1281, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1283, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1284, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1285, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1286, {low, high};
}
{
neg.f16x2 r1287, r1286;
}
{
add.f16x2 r1289, %53, %90;
}
{
add.f16x2 r1292, %60, r1289;
}
{
add.f16x2 r1295, %68, %52;
}
{
add.f16x2 r1298, r1292, r1295;
}
{
add.f16x2 r1301, %78, %67;
}
{
add.f16x2 r1304, %86, r1301;
}
{
add.f16x2 r1307, %93, %77;
}
{
add.f16x2 r1310, r1304, r1307;
}
{
add.f16x2 r1313, %53, %90;
}
{
mul.f16x2 r1316, r1313, r1281;
}
{
add.f16x2 r1319, %60, r1316;
}
{
add.f16x2 r1322, %68, %52;
}
{
mul.f16x2 r1325, r1322, r1283;
}
{
add.f16x2 r1328, r1319, r1325;
}
{
sub.f16x2 r1331, %78, %67;
}
{
mul.f16x2 r1334, r1331, r1282;
}
{
sub.f16x2 r1337, %93, %77;
}
{
mul.f16x2 r1340, r1337, r1284;
}
{
add.f16x2 r1343, r1334, r1340;
}
{
sub.f16x2 r1346, r1328, r1343;
}
{
add.f16x2 r1349, %53, %90;
}
{
mul.f16x2 r1352, r1349, r1281;
}
{
add.f16x2 r1355, %60, r1352;
}
{
add.f16x2 r1358, %68, %52;
}
{
mul.f16x2 r1361, r1358, r1283;
}
{
add.f16x2 r1364, r1355, r1361;
}
{
sub.f16x2 r1367, %78, %67;
}
{
mul.f16x2 r1370, r1367, r1282;
}
{
sub.f16x2 r1373, %93, %77;
}
{
mul.f16x2 r1376, r1373, r1284;
}
{
add.f16x2 r1379, r1370, r1376;
}
{
add.f16x2 r1382, r1364, r1379;
}
{
add.f16x2 r1385, %53, %90;
}
{
mul.f16x2 r1388, r1385, r1283;
}
{
add.f16x2 r1391, %60, r1388;
}
{
add.f16x2 r1394, %68, %52;
}
{
mul.f16x2 r1397, r1394, r1285;
}
{
add.f16x2 r1400, r1391, r1397;
}
{
sub.f16x2 r1403, %78, %67;
}
{
mul.f16x2 r1406, r1403, r1284;
}
{
sub.f16x2 r1409, %93, %77;
}
{
mul.f16x2 r1412, r1409, r1287;
}
{
add.f16x2 r1415, r1406, r1412;
}
{
sub.f16x2 r1418, r1400, r1415;
}
{
add.f16x2 r1421, %53, %90;
}
{
mul.f16x2 r1424, r1421, r1283;
}
{
add.f16x2 r1427, %60, r1424;
}
{
add.f16x2 r1430, %68, %52;
}
{
mul.f16x2 r1433, r1430, r1285;
}
{
add.f16x2 r1436, r1427, r1433;
}
{
sub.f16x2 r1439, %78, %67;
}
{
mul.f16x2 r1442, r1439, r1284;
}
{
sub.f16x2 r1445, %93, %77;
}
{
mul.f16x2 r1448, r1445, r1287;
}
{
add.f16x2 r1451, r1442, r1448;
}
{
add.f16x2 r1454, r1436, r1451;
}
{
add.f16x2 r1457, %78, %67;
}
{
mul.f16x2 r1460, r1457, r1281;
}
{
add.f16x2 r1463, %86, r1460;
}
{
add.f16x2 r1466, %93, %77;
}
{
mul.f16x2 r1469, r1466, r1283;
}
{
add.f16x2 r1472, r1463, r1469;
}
{
sub.f16x2 r1475, %53, %90;
}
{
mul.f16x2 r1478, r1475, r1282;
}
{
sub.f16x2 r1481, %68, %52;
}
{
mul.f16x2 r1484, r1481, r1284;
}
{
add.f16x2 r1487, r1478, r1484;
}
{
add.f16x2 r1490, r1472, r1487;
}
{
add.f16x2 r1493, %78, %67;
}
{
mul.f16x2 r1496, r1493, r1281;
}
{
add.f16x2 r1499, %86, r1496;
}
{
add.f16x2 r1502, %93, %77;
}
{
mul.f16x2 r1505, r1502, r1283;
}
{
add.f16x2 r1508, r1499, r1505;
}
{
sub.f16x2 r1511, %53, %90;
}
{
mul.f16x2 r1514, r1511, r1282;
}
{
sub.f16x2 r1517, %68, %52;
}
{
mul.f16x2 r1520, r1517, r1284;
}
{
add.f16x2 r1523, r1514, r1520;
}
{
sub.f16x2 r1526, r1508, r1523;
}
{
add.f16x2 r1529, %78, %67;
}
{
mul.f16x2 r1532, r1529, r1283;
}
{
add.f16x2 r1535, %86, r1532;
}
{
add.f16x2 r1538, %93, %77;
}
{
mul.f16x2 r1541, r1538, r1285;
}
{
add.f16x2 r1544, r1535, r1541;
}
{
sub.f16x2 r1547, %53, %90;
}
{
mul.f16x2 r1550, r1547, r1284;
}
{
sub.f16x2 r1553, %68, %52;
}
{
mul.f16x2 r1556, r1553, r1287;
}
{
add.f16x2 r1559, r1550, r1556;
}
{
add.f16x2 r1562, r1544, r1559;
}
{
add.f16x2 r1565, %78, %67;
}
{
mul.f16x2 r1568, r1565, r1283;
}
{
add.f16x2 r1571, %86, r1568;
}
{
add.f16x2 r1574, %93, %77;
}
{
mul.f16x2 r1577, r1574, r1285;
}
{
add.f16x2 r1580, r1571, r1577;
}
{
sub.f16x2 r1583, %53, %90;
}
{
mul.f16x2 r1586, r1583, r1284;
}
{
sub.f16x2 r1589, %68, %52;
}
{
mul.f16x2 r1592, r1589, r1287;
}
{
add.f16x2 r1595, r1586, r1592;
}
{
sub.f16x2 r1598, r1580, r1595;
}
mov.f32 f330, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1601, {low, high};
}
mov.f32 f332, 0fBE7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r1602, {low, high};
}
mov.f32 f334, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r1603, {low, high};
}
mov.f32 f336, 0fBEF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r1604, {low, high};
}
mov.f32 f338, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r1605, {low, high};
}
mov.f32 f340, 0fBF2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1606, {low, high};
}
mov.f32 f342, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r1607, {low, high};
}
mov.f32 f344, 0fBF5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r1608, {low, high};
}
mov.f32 f350, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r1611, {low, high};
}
mov.f32 f352, 0fBF7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1612, {low, high};
}
mov.f32 f358, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r1615, {low, high};
}
mov.f32 f360, 0fBF67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r1616, {low, high};
}
mov.f32 f390, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1617, {low, high};
}
mov.f32 f364, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r1618, {low, high};
}
mov.f32 f374, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1623, {low, high};
}
mov.f32 f376, 0fBE00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1631, {low, high};
}
mov.f32 f392, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r1632, {low, high};
}
{
mul.f16x2 r1649, r386, r1601;
}
{
mul.f16x2 r1652, r530, r1602;
}
{
sub.f16x2 r1655, r1649, r1652;
}
{
mul.f16x2 r1658, r386, r1602;
}
{
fma.rn.f16x2 r1661, r530, r1601, r1658;
}
{
mul.f16x2 r1665, r706, r1603;
}
{
mul.f16x2 r1668, r850, r1604;
}
{
sub.f16x2 r1671, r1665, r1668;
}
{
mul.f16x2 r1674, r706, r1604;
}
{
fma.rn.f16x2 r1677, r850, r1603, r1674;
}
{
mul.f16x2 r1681, r1026, r1605;
}
{
mul.f16x2 r1684, r1170, r1606;
}
{
sub.f16x2 r1687, r1681, r1684;
}
{
mul.f16x2 r1690, r1026, r1606;
}
{
fma.rn.f16x2 r1693, r1170, r1605, r1690;
}
{
mul.f16x2 r1697, r1346, r1607;
}
{
mul.f16x2 r1700, r1490, r1608;
}
{
sub.f16x2 r1703, r1697, r1700;
}
{
mul.f16x2 r1706, r1346, r1608;
}
{
fma.rn.f16x2 r1709, r1490, r1607, r1706;
}
{
mul.f16x2 r1713, r458, r1603;
}
{
mul.f16x2 r1716, r602, r1604;
}
{
sub.f16x2 r1719, r1713, r1716;
}
{
mul.f16x2 r1722, r458, r1604;
}
{
fma.rn.f16x2 r1725, r602, r1603, r1722;
}
{
mul.f16x2 r1729, r778, r1607;
}
{
mul.f16x2 r1732, r922, r1608;
}
{
sub.f16x2 r1735, r1729, r1732;
}
{
mul.f16x2 r1738, r778, r1608;
}
{
fma.rn.f16x2 r1741, r922, r1607, r1738;
}
{
mul.f16x2 r1745, r1098, r1611;
}
{
mul.f16x2 r1748, r1242, r1612;
}
{
sub.f16x2 r1751, r1745, r1748;
}
{
mul.f16x2 r1754, r1098, r1612;
}
{
fma.rn.f16x2 r1757, r1242, r1611, r1754;
}
{
mul.f16x2 r1761, r1418, r1615;
}
{
mul.f16x2 r1764, r1562, r1616;
}
{
sub.f16x2 r1767, r1761, r1764;
}
{
mul.f16x2 r1770, r1418, r1616;
}
{
fma.rn.f16x2 r1773, r1562, r1615, r1770;
}
{
mul.f16x2 r1777, r494, r1605;
}
{
mul.f16x2 r1780, r638, r1606;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r494, r1606;
}
{
fma.rn.f16x2 r1789, r638, r1605, r1786;
}
{
mul.f16x2 r1793, r814, r1611;
}
{
mul.f16x2 r1796, r958, r1612;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r814, r1612;
}
{
fma.rn.f16x2 r1805, r958, r1611, r1802;
}
{
mul.f16x2 r1809, r1134, r1617;
}
{
mul.f16x2 r1812, r1278, r1618;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r1134, r1618;
}
{
fma.rn.f16x2 r1821, r1278, r1617, r1818;
}
{
mul.f16x2 r1825, r1454, r1623;
}
{
mul.f16x2 r1828, r1598, r1624;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r1454, r1624;
}
{
fma.rn.f16x2 r1837, r1598, r1623, r1834;
}
{
mul.f16x2 r1841, r422, r1607;
}
{
mul.f16x2 r1844, r566, r1608;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r422, r1608;
}
{
fma.rn.f16x2 r1853, r566, r1607, r1850;
}
{
mul.f16x2 r1857, r742, r1615;
}
{
mul.f16x2 r1860, r886, r1616;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r742, r1616;
}
{
fma.rn.f16x2 r1869, r886, r1615, r1866;
}
{
mul.f16x2 r1873, r1062, r1623;
}
{
mul.f16x2 r1876, r1206, r1624;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r1062, r1624;
}
{
fma.rn.f16x2 r1885, r1206, r1623, r1882;
}
{
mul.f16x2 r1889, r1382, r1631;
}
{
mul.f16x2 r1892, r1526, r1632;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r1382, r1632;
}
{
fma.rn.f16x2 r1901, r1526, r1631, r1898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1905, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1906, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1909, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1910, {low, high};
}
{
neg.f16x2 r1911, r1910;
}
{
add.f16x2 r1913, r338, r1298;
}
{
add.f16x2 r1916, r18, r1913;
}
{
add.f16x2 r1919, r658, r978;
}
{
add.f16x2 r1922, r1916, r1919;
}
{
add.f16x2 r1925, r350, r1310;
}
{
add.f16x2 r1928, r30, r1925;
}
{
add.f16x2 r1931, r670, r990;
}
{
add.f16x2 r1934, r1928, r1931;
}
{
add.f16x2 r1937, r338, r1298;
}
{
mul.f16x2 r1940, r1937, r1905;
}
{
add.f16x2 r1943, r18, r1940;
}
{
add.f16x2 r1946, r658, r978;
}
{
mul.f16x2 r1949, r1946, r1907;
}
{
add.f16x2 r1952, r1943, r1949;
}
{
sub.f16x2 r1955, r350, r1310;
}
{
mul.f16x2 r1958, r1955, r1906;
}
{
sub.f16x2 r1961, r670, r990;
}
{
mul.f16x2 r1964, r1961, r1908;
}
{
add.f16x2 r1967, r1958, r1964;
}
{
sub.f16x2 r1970, r1952, r1967;
}
{
add.f16x2 r1973, r338, r1298;
}
{
mul.f16x2 r1976, r1973, r1905;
}
{
add.f16x2 r1979, r18, r1976;
}
{
add.f16x2 r1982, r658, r978;
}
{
mul.f16x2 r1985, r1982, r1907;
}
{
add.f16x2 r1988, r1979, r1985;
}
{
sub.f16x2 r1991, r350, r1310;
}
{
mul.f16x2 r1994, r1991, r1906;
}
{
sub.f16x2 r1997, r670, r990;
}
{
mul.f16x2 r2000, r1997, r1908;
}
{
add.f16x2 r2003, r1994, r2000;
}
{
add.f16x2 r2006, r1988, r2003;
}
{
add.f16x2 r2009, r338, r1298;
}
{
mul.f16x2 r2012, r2009, r1907;
}
{
add.f16x2 r2015, r18, r2012;
}
{
add.f16x2 r2018, r658, r978;
}
{
mul.f16x2 r2021, r2018, r1909;
}
{
add.f16x2 r2024, r2015, r2021;
}
{
sub.f16x2 r2027, r350, r1310;
}
{
mul.f16x2 r2030, r2027, r1908;
}
{
sub.f16x2 r2033, r670, r990;
}
{
mul.f16x2 r2036, r2033, r1911;
}
{
add.f16x2 r2039, r2030, r2036;
}
{
sub.f16x2 r2042, r2024, r2039;
}
{
add.f16x2 r2045, r338, r1298;
}
{
mul.f16x2 r2048, r2045, r1907;
}
{
add.f16x2 r2051, r18, r2048;
}
{
add.f16x2 r2054, r658, r978;
}
{
mul.f16x2 r2057, r2054, r1909;
}
{
add.f16x2 r2060, r2051, r2057;
}
{
sub.f16x2 r2063, r350, r1310;
}
{
mul.f16x2 r2066, r2063, r1908;
}
{
sub.f16x2 r2069, r670, r990;
}
{
mul.f16x2 r2072, r2069, r1911;
}
{
add.f16x2 r2075, r2066, r2072;
}
{
add.f16x2 r2078, r2060, r2075;
}
{
add.f16x2 r2081, r350, r1310;
}
{
mul.f16x2 r2084, r2081, r1905;
}
{
add.f16x2 r2087, r30, r2084;
}
{
add.f16x2 r2090, r670, r990;
}
{
mul.f16x2 r2093, r2090, r1907;
}
{
add.f16x2 r2096, r2087, r2093;
}
{
sub.f16x2 r2099, r338, r1298;
}
{
mul.f16x2 r2102, r2099, r1906;
}
{
sub.f16x2 r2105, r658, r978;
}
{
mul.f16x2 r2108, r2105, r1908;
}
{
add.f16x2 r2111, r2102, r2108;
}
{
add.f16x2 r2114, r2096, r2111;
}
{
add.f16x2 r2117, r350, r1310;
}
{
mul.f16x2 r2120, r2117, r1905;
}
{
add.f16x2 r2123, r30, r2120;
}
{
add.f16x2 r2126, r670, r990;
}
{
mul.f16x2 r2129, r2126, r1907;
}
{
add.f16x2 r2132, r2123, r2129;
}
{
sub.f16x2 r2135, r338, r1298;
}
{
mul.f16x2 r2138, r2135, r1906;
}
{
sub.f16x2 r2141, r658, r978;
}
{
mul.f16x2 r2144, r2141, r1908;
}
{
add.f16x2 r2147, r2138, r2144;
}
{
sub.f16x2 r2150, r2132, r2147;
}
{
add.f16x2 r2153, r350, r1310;
}
{
mul.f16x2 r2156, r2153, r1907;
}
{
add.f16x2 r2159, r30, r2156;
}
{
add.f16x2 r2162, r670, r990;
}
{
mul.f16x2 r2165, r2162, r1909;
}
{
add.f16x2 r2168, r2159, r2165;
}
{
sub.f16x2 r2171, r338, r1298;
}
{
mul.f16x2 r2174, r2171, r1908;
}
{
sub.f16x2 r2177, r658, r978;
}
{
mul.f16x2 r2180, r2177, r1911;
}
{
add.f16x2 r2183, r2174, r2180;
}
{
add.f16x2 r2186, r2168, r2183;
}
{
add.f16x2 r2189, r350, r1310;
}
{
mul.f16x2 r2192, r2189, r1907;
}
{
add.f16x2 r2195, r30, r2192;
}
{
add.f16x2 r2198, r670, r990;
}
{
mul.f16x2 r2201, r2198, r1909;
}
{
add.f16x2 r2204, r2195, r2201;
}
{
sub.f16x2 r2207, r338, r1298;
}
{
mul.f16x2 r2210, r2207, r1908;
}
{
sub.f16x2 r2213, r658, r978;
}
{
mul.f16x2 r2216, r2213, r1911;
}
{
add.f16x2 r2219, r2210, r2216;
}
{
sub.f16x2 r2222, r2204, r2219;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2226, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2227, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2228, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2230, {low, high};
}
{
neg.f16x2 r2231, r2230;
}
{
add.f16x2 r2233, r1655, r1703;
}
{
add.f16x2 r2236, r66, r2233;
}
{
add.f16x2 r2239, r1671, r1687;
}
{
add.f16x2 r2242, r2236, r2239;
}
{
add.f16x2 r2245, r1661, r1709;
}
{
add.f16x2 r2248, r210, r2245;
}
{
add.f16x2 r2251, r1677, r1693;
}
{
add.f16x2 r2254, r2248, r2251;
}
{
add.f16x2 r2257, r1655, r1703;
}
{
mul.f16x2 r2260, r2257, r2225;
}
{
add.f16x2 r2263, r66, r2260;
}
{
add.f16x2 r2266, r1671, r1687;
}
{
mul.f16x2 r2269, r2266, r2227;
}
{
add.f16x2 r2272, r2263, r2269;
}
{
sub.f16x2 r2275, r1661, r1709;
}
{
mul.f16x2 r2278, r2275, r2226;
}
{
sub.f16x2 r2281, r1677, r1693;
}
{
mul.f16x2 r2284, r2281, r2228;
}
{
add.f16x2 r2287, r2278, r2284;
}
{
sub.f16x2 r2290, r2272, r2287;
}
{
add.f16x2 r2293, r1655, r1703;
}
{
mul.f16x2 r2296, r2293, r2225;
}
{
add.f16x2 r2299, r66, r2296;
}
{
add.f16x2 r2302, r1671, r1687;
}
{
mul.f16x2 r2305, r2302, r2227;
}
{
add.f16x2 r2308, r2299, r2305;
}
{
sub.f16x2 r2311, r1661, r1709;
}
{
mul.f16x2 r2314, r2311, r2226;
}
{
sub.f16x2 r2317, r1677, r1693;
}
{
mul.f16x2 r2320, r2317, r2228;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
add.f16x2 r2326, r2308, r2323;
}
{
add.f16x2 r2329, r1655, r1703;
}
{
mul.f16x2 r2332, r2329, r2227;
}
{
add.f16x2 r2335, r66, r2332;
}
{
add.f16x2 r2338, r1671, r1687;
}
{
mul.f16x2 r2341, r2338, r2229;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
sub.f16x2 r2347, r1661, r1709;
}
{
mul.f16x2 r2350, r2347, r2228;
}
{
sub.f16x2 r2353, r1677, r1693;
}
{
mul.f16x2 r2356, r2353, r2231;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
sub.f16x2 r2362, r2344, r2359;
}
{
add.f16x2 r2365, r1655, r1703;
}
{
mul.f16x2 r2368, r2365, r2227;
}
{
add.f16x2 r2371, r66, r2368;
}
{
add.f16x2 r2374, r1671, r1687;
}
{
mul.f16x2 r2377, r2374, r2229;
}
{
add.f16x2 r2380, r2371, r2377;
}
{
sub.f16x2 r2383, r1661, r1709;
}
{
mul.f16x2 r2386, r2383, r2228;
}
{
sub.f16x2 r2389, r1677, r1693;
}
{
mul.f16x2 r2392, r2389, r2231;
}
{
add.f16x2 r2395, r2386, r2392;
}
{
add.f16x2 r2398, r2380, r2395;
}
{
add.f16x2 r2401, r1661, r1709;
}
{
mul.f16x2 r2404, r2401, r2225;
}
{
add.f16x2 r2407, r210, r2404;
}
{
add.f16x2 r2410, r1677, r1693;
}
{
mul.f16x2 r2413, r2410, r2227;
}
{
add.f16x2 r2416, r2407, r2413;
}
{
sub.f16x2 r2419, r1655, r1703;
}
{
mul.f16x2 r2422, r2419, r2226;
}
{
sub.f16x2 r2425, r1671, r1687;
}
{
mul.f16x2 r2428, r2425, r2228;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
add.f16x2 r2434, r2416, r2431;
}
{
add.f16x2 r2437, r1661, r1709;
}
{
mul.f16x2 r2440, r2437, r2225;
}
{
add.f16x2 r2443, r210, r2440;
}
{
add.f16x2 r2446, r1677, r1693;
}
{
mul.f16x2 r2449, r2446, r2227;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
sub.f16x2 r2455, r1655, r1703;
}
{
mul.f16x2 r2458, r2455, r2226;
}
{
sub.f16x2 r2461, r1671, r1687;
}
{
mul.f16x2 r2464, r2461, r2228;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
sub.f16x2 r2470, r2452, r2467;
}
{
add.f16x2 r2473, r1661, r1709;
}
{
mul.f16x2 r2476, r2473, r2227;
}
{
add.f16x2 r2479, r210, r2476;
}
{
add.f16x2 r2482, r1677, r1693;
}
{
mul.f16x2 r2485, r2482, r2229;
}
{
add.f16x2 r2488, r2479, r2485;
}
{
sub.f16x2 r2491, r1655, r1703;
}
{
mul.f16x2 r2494, r2491, r2228;
}
{
sub.f16x2 r2497, r1671, r1687;
}
{
mul.f16x2 r2500, r2497, r2231;
}
{
add.f16x2 r2503, r2494, r2500;
}
{
add.f16x2 r2506, r2488, r2503;
}
{
add.f16x2 r2509, r1661, r1709;
}
{
mul.f16x2 r2512, r2509, r2227;
}
{
add.f16x2 r2515, r210, r2512;
}
{
add.f16x2 r2518, r1677, r1693;
}
{
mul.f16x2 r2521, r2518, r2229;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
sub.f16x2 r2527, r1655, r1703;
}
{
mul.f16x2 r2530, r2527, r2228;
}
{
sub.f16x2 r2533, r1671, r1687;
}
{
mul.f16x2 r2536, r2533, r2231;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
sub.f16x2 r2542, r2524, r2539;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2545, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2546, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2547, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2548, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2549, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2550, {low, high};
}
{
neg.f16x2 r2551, r2550;
}
{
add.f16x2 r2553, r1719, r1767;
}
{
add.f16x2 r2556, r138, r2553;
}
{
add.f16x2 r2559, r1735, r1751;
}
{
add.f16x2 r2562, r2556, r2559;
}
{
add.f16x2 r2565, r1725, r1773;
}
{
add.f16x2 r2568, r282, r2565;
}
{
add.f16x2 r2571, r1741, r1757;
}
{
add.f16x2 r2574, r2568, r2571;
}
{
add.f16x2 r2577, r1719, r1767;
}
{
mul.f16x2 r2580, r2577, r2545;
}
{
add.f16x2 r2583, r138, r2580;
}
{
add.f16x2 r2586, r1735, r1751;
}
{
mul.f16x2 r2589, r2586, r2547;
}
{
add.f16x2 r2592, r2583, r2589;
}
{
sub.f16x2 r2595, r1725, r1773;
}
{
mul.f16x2 r2598, r2595, r2546;
}
{
sub.f16x2 r2601, r1741, r1757;
}
{
mul.f16x2 r2604, r2601, r2548;
}
{
add.f16x2 r2607, r2598, r2604;
}
{
sub.f16x2 r2610, r2592, r2607;
}
{
add.f16x2 r2613, r1719, r1767;
}
{
mul.f16x2 r2616, r2613, r2545;
}
{
add.f16x2 r2619, r138, r2616;
}
{
add.f16x2 r2622, r1735, r1751;
}
{
mul.f16x2 r2625, r2622, r2547;
}
{
add.f16x2 r2628, r2619, r2625;
}
{
sub.f16x2 r2631, r1725, r1773;
}
{
mul.f16x2 r2634, r2631, r2546;
}
{
sub.f16x2 r2637, r1741, r1757;
}
{
mul.f16x2 r2640, r2637, r2548;
}
{
add.f16x2 r2643, r2634, r2640;
}
{
add.f16x2 r2646, r2628, r2643;
}
{
add.f16x2 r2649, r1719, r1767;
}
{
mul.f16x2 r2652, r2649, r2547;
}
{
add.f16x2 r2655, r138, r2652;
}
{
add.f16x2 r2658, r1735, r1751;
}
{
mul.f16x2 r2661, r2658, r2549;
}
{
add.f16x2 r2664, r2655, r2661;
}
{
sub.f16x2 r2667, r1725, r1773;
}
{
mul.f16x2 r2670, r2667, r2548;
}
{
sub.f16x2 r2673, r1741, r1757;
}
{
mul.f16x2 r2676, r2673, r2551;
}
{
add.f16x2 r2679, r2670, r2676;
}
{
sub.f16x2 r2682, r2664, r2679;
}
{
add.f16x2 r2685, r1719, r1767;
}
{
mul.f16x2 r2688, r2685, r2547;
}
{
add.f16x2 r2691, r138, r2688;
}
{
add.f16x2 r2694, r1735, r1751;
}
{
mul.f16x2 r2697, r2694, r2549;
}
{
add.f16x2 r2700, r2691, r2697;
}
{
sub.f16x2 r2703, r1725, r1773;
}
{
mul.f16x2 r2706, r2703, r2548;
}
{
sub.f16x2 r2709, r1741, r1757;
}
{
mul.f16x2 r2712, r2709, r2551;
}
{
add.f16x2 r2715, r2706, r2712;
}
{
add.f16x2 r2718, r2700, r2715;
}
{
add.f16x2 r2721, r1725, r1773;
}
{
mul.f16x2 r2724, r2721, r2545;
}
{
add.f16x2 r2727, r282, r2724;
}
{
add.f16x2 r2730, r1741, r1757;
}
{
mul.f16x2 r2733, r2730, r2547;
}
{
add.f16x2 r2736, r2727, r2733;
}
{
sub.f16x2 r2739, r1719, r1767;
}
{
mul.f16x2 r2742, r2739, r2546;
}
{
sub.f16x2 r2745, r1735, r1751;
}
{
mul.f16x2 r2748, r2745, r2548;
}
{
add.f16x2 r2751, r2742, r2748;
}
{
add.f16x2 r2754, r2736, r2751;
}
{
add.f16x2 r2757, r1725, r1773;
}
{
mul.f16x2 r2760, r2757, r2545;
}
{
add.f16x2 r2763, r282, r2760;
}
{
add.f16x2 r2766, r1741, r1757;
}
{
mul.f16x2 r2769, r2766, r2547;
}
{
add.f16x2 r2772, r2763, r2769;
}
{
sub.f16x2 r2775, r1719, r1767;
}
{
mul.f16x2 r2778, r2775, r2546;
}
{
sub.f16x2 r2781, r1735, r1751;
}
{
mul.f16x2 r2784, r2781, r2548;
}
{
add.f16x2 r2787, r2778, r2784;
}
{
sub.f16x2 r2790, r2772, r2787;
}
{
add.f16x2 r2793, r1725, r1773;
}
{
mul.f16x2 r2796, r2793, r2547;
}
{
add.f16x2 r2799, r282, r2796;
}
{
add.f16x2 r2802, r1741, r1757;
}
{
mul.f16x2 r2805, r2802, r2549;
}
{
add.f16x2 r2808, r2799, r2805;
}
{
sub.f16x2 r2811, r1719, r1767;
}
{
mul.f16x2 r2814, r2811, r2548;
}
{
sub.f16x2 r2817, r1735, r1751;
}
{
mul.f16x2 r2820, r2817, r2551;
}
{
add.f16x2 r2823, r2814, r2820;
}
{
add.f16x2 r2826, r2808, r2823;
}
{
add.f16x2 r2829, r1725, r1773;
}
{
mul.f16x2 r2832, r2829, r2547;
}
{
add.f16x2 r2835, r282, r2832;
}
{
add.f16x2 r2838, r1741, r1757;
}
{
mul.f16x2 r2841, r2838, r2549;
}
{
add.f16x2 r2844, r2835, r2841;
}
{
sub.f16x2 r2847, r1719, r1767;
}
{
mul.f16x2 r2850, r2847, r2548;
}
{
sub.f16x2 r2853, r1735, r1751;
}
{
mul.f16x2 r2856, r2853, r2551;
}
{
add.f16x2 r2859, r2850, r2856;
}
{
sub.f16x2 r2862, r2844, r2859;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2866, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2868, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2869, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2870, {low, high};
}
{
neg.f16x2 r2871, r2870;
}
{
add.f16x2 r2873, r1783, r1831;
}
{
add.f16x2 r2876, r174, r2873;
}
{
add.f16x2 r2879, r1799, r1815;
}
{
add.f16x2 r2882, r2876, r2879;
}
{
add.f16x2 r2885, r1789, r1837;
}
{
add.f16x2 r2888, r318, r2885;
}
{
add.f16x2 r2891, r1805, r1821;
}
{
add.f16x2 r2894, r2888, r2891;
}
{
add.f16x2 r2897, r1783, r1831;
}
{
mul.f16x2 r2900, r2897, r2865;
}
{
add.f16x2 r2903, r174, r2900;
}
{
add.f16x2 r2906, r1799, r1815;
}
{
mul.f16x2 r2909, r2906, r2867;
}
{
add.f16x2 r2912, r2903, r2909;
}
{
sub.f16x2 r2915, r1789, r1837;
}
{
mul.f16x2 r2918, r2915, r2866;
}
{
sub.f16x2 r2921, r1805, r1821;
}
{
mul.f16x2 r2924, r2921, r2868;
}
{
add.f16x2 r2927, r2918, r2924;
}
{
sub.f16x2 r2930, r2912, r2927;
}
{
add.f16x2 r2933, r1783, r1831;
}
{
mul.f16x2 r2936, r2933, r2865;
}
{
add.f16x2 r2939, r174, r2936;
}
{
add.f16x2 r2942, r1799, r1815;
}
{
mul.f16x2 r2945, r2942, r2867;
}
{
add.f16x2 r2948, r2939, r2945;
}
{
sub.f16x2 r2951, r1789, r1837;
}
{
mul.f16x2 r2954, r2951, r2866;
}
{
sub.f16x2 r2957, r1805, r1821;
}
{
mul.f16x2 r2960, r2957, r2868;
}
{
add.f16x2 r2963, r2954, r2960;
}
{
add.f16x2 r2966, r2948, r2963;
}
{
add.f16x2 r2969, r1783, r1831;
}
{
mul.f16x2 r2972, r2969, r2867;
}
{
add.f16x2 r2975, r174, r2972;
}
{
add.f16x2 r2978, r1799, r1815;
}
{
mul.f16x2 r2981, r2978, r2869;
}
{
add.f16x2 r2984, r2975, r2981;
}
{
sub.f16x2 r2987, r1789, r1837;
}
{
mul.f16x2 r2990, r2987, r2868;
}
{
sub.f16x2 r2993, r1805, r1821;
}
{
mul.f16x2 r2996, r2993, r2871;
}
{
add.f16x2 r2999, r2990, r2996;
}
{
sub.f16x2 r3002, r2984, r2999;
}
{
add.f16x2 r3005, r1783, r1831;
}
{
mul.f16x2 r3008, r3005, r2867;
}
{
add.f16x2 r3011, r174, r3008;
}
{
add.f16x2 r3014, r1799, r1815;
}
{
mul.f16x2 r3017, r3014, r2869;
}
{
add.f16x2 r3020, r3011, r3017;
}
{
sub.f16x2 r3023, r1789, r1837;
}
{
mul.f16x2 r3026, r3023, r2868;
}
{
sub.f16x2 r3029, r1805, r1821;
}
{
mul.f16x2 r3032, r3029, r2871;
}
{
add.f16x2 r3035, r3026, r3032;
}
{
add.f16x2 r3038, r3020, r3035;
}
{
add.f16x2 r3041, r1789, r1837;
}
{
mul.f16x2 r3044, r3041, r2865;
}
{
add.f16x2 r3047, r318, r3044;
}
{
add.f16x2 r3050, r1805, r1821;
}
{
mul.f16x2 r3053, r3050, r2867;
}
{
add.f16x2 r3056, r3047, r3053;
}
{
sub.f16x2 r3059, r1783, r1831;
}
{
mul.f16x2 r3062, r3059, r2866;
}
{
sub.f16x2 r3065, r1799, r1815;
}
{
mul.f16x2 r3068, r3065, r2868;
}
{
add.f16x2 r3071, r3062, r3068;
}
{
add.f16x2 r3074, r3056, r3071;
}
{
add.f16x2 r3077, r1789, r1837;
}
{
mul.f16x2 r3080, r3077, r2865;
}
{
add.f16x2 r3083, r318, r3080;
}
{
add.f16x2 r3086, r1805, r1821;
}
{
mul.f16x2 r3089, r3086, r2867;
}
{
add.f16x2 r3092, r3083, r3089;
}
{
sub.f16x2 r3095, r1783, r1831;
}
{
mul.f16x2 r3098, r3095, r2866;
}
{
sub.f16x2 r3101, r1799, r1815;
}
{
mul.f16x2 r3104, r3101, r2868;
}
{
add.f16x2 r3107, r3098, r3104;
}
{
sub.f16x2 r3110, r3092, r3107;
}
{
add.f16x2 r3113, r1789, r1837;
}
{
mul.f16x2 r3116, r3113, r2867;
}
{
add.f16x2 r3119, r318, r3116;
}
{
add.f16x2 r3122, r1805, r1821;
}
{
mul.f16x2 r3125, r3122, r2869;
}
{
add.f16x2 r3128, r3119, r3125;
}
{
sub.f16x2 r3131, r1783, r1831;
}
{
mul.f16x2 r3134, r3131, r2868;
}
{
sub.f16x2 r3137, r1799, r1815;
}
{
mul.f16x2 r3140, r3137, r2871;
}
{
add.f16x2 r3143, r3134, r3140;
}
{
add.f16x2 r3146, r3128, r3143;
}
{
add.f16x2 r3149, r1789, r1837;
}
{
mul.f16x2 r3152, r3149, r2867;
}
{
add.f16x2 r3155, r318, r3152;
}
{
add.f16x2 r3158, r1805, r1821;
}
{
mul.f16x2 r3161, r3158, r2869;
}
{
add.f16x2 r3164, r3155, r3161;
}
{
sub.f16x2 r3167, r1783, r1831;
}
{
mul.f16x2 r3170, r3167, r2868;
}
{
sub.f16x2 r3173, r1799, r1815;
}
{
mul.f16x2 r3176, r3173, r2871;
}
{
add.f16x2 r3179, r3170, r3176;
}
{
sub.f16x2 r3182, r3164, r3179;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3185, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3186, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r3188, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3189, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3190, {low, high};
}
{
neg.f16x2 r3191, r3190;
}
{
add.f16x2 r3193, r1847, r1895;
}
{
add.f16x2 r3196, r102, r3193;
}
{
add.f16x2 r3199, r1863, r1879;
}
{
add.f16x2 r3202, r3196, r3199;
}
{
add.f16x2 r3205, r1853, r1901;
}
{
add.f16x2 r3208, r246, r3205;
}
{
add.f16x2 r3211, r1869, r1885;
}
{
add.f16x2 r3214, r3208, r3211;
}
{
add.f16x2 r3217, r1847, r1895;
}
{
mul.f16x2 r3220, r3217, r3185;
}
{
add.f16x2 r3223, r102, r3220;
}
{
add.f16x2 r3226, r1863, r1879;
}
{
mul.f16x2 r3229, r3226, r3187;
}
{
add.f16x2 r3232, r3223, r3229;
}
{
sub.f16x2 r3235, r1853, r1901;
}
{
mul.f16x2 r3238, r3235, r3186;
}
{
sub.f16x2 r3241, r1869, r1885;
}
{
mul.f16x2 r3244, r3241, r3188;
}
{
add.f16x2 r3247, r3238, r3244;
}
{
sub.f16x2 r3250, r3232, r3247;
}
{
add.f16x2 r3253, r1847, r1895;
}
{
mul.f16x2 r3256, r3253, r3185;
}
{
add.f16x2 r3259, r102, r3256;
}
{
add.f16x2 r3262, r1863, r1879;
}
{
mul.f16x2 r3265, r3262, r3187;
}
{
add.f16x2 r3268, r3259, r3265;
}
{
sub.f16x2 r3271, r1853, r1901;
}
{
mul.f16x2 r3274, r3271, r3186;
}
{
sub.f16x2 r3277, r1869, r1885;
}
{
mul.f16x2 r3280, r3277, r3188;
}
{
add.f16x2 r3283, r3274, r3280;
}
{
add.f16x2 r3286, r3268, r3283;
}
{
add.f16x2 r3289, r1847, r1895;
}
{
mul.f16x2 r3292, r3289, r3187;
}
{
add.f16x2 r3295, r102, r3292;
}
{
add.f16x2 r3298, r1863, r1879;
}
{
mul.f16x2 r3301, r3298, r3189;
}
{
add.f16x2 r3304, r3295, r3301;
}
{
sub.f16x2 r3307, r1853, r1901;
}
{
mul.f16x2 r3310, r3307, r3188;
}
{
sub.f16x2 r3313, r1869, r1885;
}
{
mul.f16x2 r3316, r3313, r3191;
}
{
add.f16x2 r3319, r3310, r3316;
}
{
sub.f16x2 r3322, r3304, r3319;
}
{
add.f16x2 r3325, r1847, r1895;
}
{
mul.f16x2 r3328, r3325, r3187;
}
{
add.f16x2 r3331, r102, r3328;
}
{
add.f16x2 r3334, r1863, r1879;
}
{
mul.f16x2 r3337, r3334, r3189;
}
{
add.f16x2 r3340, r3331, r3337;
}
{
sub.f16x2 r3343, r1853, r1901;
}
{
mul.f16x2 r3346, r3343, r3188;
}
{
sub.f16x2 r3349, r1869, r1885;
}
{
mul.f16x2 r3352, r3349, r3191;
}
{
add.f16x2 r3355, r3346, r3352;
}
{
add.f16x2 r3358, r3340, r3355;
}
{
add.f16x2 r3361, r1853, r1901;
}
{
mul.f16x2 r3364, r3361, r3185;
}
{
add.f16x2 r3367, r246, r3364;
}
{
add.f16x2 r3370, r1869, r1885;
}
{
mul.f16x2 r3373, r3370, r3187;
}
{
add.f16x2 r3376, r3367, r3373;
}
{
sub.f16x2 r3379, r1847, r1895;
}
{
mul.f16x2 r3382, r3379, r3186;
}
{
sub.f16x2 r3385, r1863, r1879;
}
{
mul.f16x2 r3388, r3385, r3188;
}
{
add.f16x2 r3391, r3382, r3388;
}
{
add.f16x2 r3394, r3376, r3391;
}
{
add.f16x2 r3397, r1853, r1901;
}
{
mul.f16x2 r3400, r3397, r3185;
}
{
add.f16x2 r3403, r246, r3400;
}
{
add.f16x2 r3406, r1869, r1885;
}
{
mul.f16x2 r3409, r3406, r3187;
}
{
add.f16x2 r3412, r3403, r3409;
}
{
sub.f16x2 r3415, r1847, r1895;
}
{
mul.f16x2 r3418, r3415, r3186;
}
{
sub.f16x2 r3421, r1863, r1879;
}
{
mul.f16x2 r3424, r3421, r3188;
}
{
add.f16x2 r3427, r3418, r3424;
}
{
sub.f16x2 r3430, r3412, r3427;
}
{
add.f16x2 r3433, r1853, r1901;
}
{
mul.f16x2 r3436, r3433, r3187;
}
{
add.f16x2 r3439, r246, r3436;
}
{
add.f16x2 r3442, r1869, r1885;
}
{
mul.f16x2 r3445, r3442, r3189;
}
{
add.f16x2 r3448, r3439, r3445;
}
{
sub.f16x2 r3451, r1847, r1895;
}
{
mul.f16x2 r3454, r3451, r3188;
}
{
sub.f16x2 r3457, r1863, r1879;
}
{
mul.f16x2 r3460, r3457, r3191;
}
{
add.f16x2 r3463, r3454, r3460;
}
{
add.f16x2 r3466, r3448, r3463;
}
{
add.f16x2 r3469, r1853, r1901;
}
{
mul.f16x2 r3472, r3469, r3187;
}
{
add.f16x2 r3475, r246, r3472;
}
{
add.f16x2 r3478, r1869, r1885;
}
{
mul.f16x2 r3481, r3478, r3189;
}
{
add.f16x2 r3484, r3475, r3481;
}
{
sub.f16x2 r3487, r1847, r1895;
}
{
mul.f16x2 r3490, r3487, r3188;
}
{
sub.f16x2 r3493, r1863, r1879;
}
{
mul.f16x2 r3496, r3493, r3191;
}
{
add.f16x2 r3499, r3490, r3496;
}
{
sub.f16x2 r3502, r3484, r3499;
}
mul.wide.u32 rd2, r10394, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r10395, rd3;
mul.lo.s32 r10396, r10395, 125;
sub.s32 r10397, r10394, r10396;
cvt.rn.f32.u32 f597, r10397;
mul.f32 f598, f597, 0f3B03C498;
cos.approx.f32 f217, f598;
sin.approx.f32 f599, f598;
neg.f32 f218, f599;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r3505, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3508, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3510, {high, high};
}
{
mul.f16x2 r3512, r2254, r3510;
}
{
neg.f16x2 r3515, r3512;
}
{
fma.rn.f16x2 r3517, r2242, r3508, r3515;
}
{
mul.f16x2 r3521, r2242, r3510;
}
{
fma.rn.f16x2 r3524, r2254, r3508, r3521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3528, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3530, {high, high};
}
mov.f32 f533, 0fBF800000;
mov.f32 f534, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3532, {low, high};
}
{
mul.f16x2 r3533, r3530, r3532;
}
{
mul.f16x2 r3536, r3505, r3528;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3539, {high, low};
}
{
fma.rn.f16x2 r3541, r3533, r3539, r3536;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3545, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3547, {high, high};
}
{
mul.f16x2 r3549, r2574, r3547;
}
{
neg.f16x2 r3552, r3549;
}
{
fma.rn.f16x2 r3554, r2562, r3545, r3552;
}
{
mul.f16x2 r3558, r2562, r3547;
}
{
fma.rn.f16x2 r3561, r2574, r3545, r3558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3565, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3567, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3569, {low, high};
}
{
mul.f16x2 r3570, r3567, r3569;
}
{
mul.f16x2 r3573, r3541, r3565;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3576, {high, low};
}
{
fma.rn.f16x2 r3578, r3570, r3576, r3573;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3582, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3584, {high, high};
}
{
mul.f16x2 r3586, r2894, r3584;
}
{
neg.f16x2 r3589, r3586;
}
{
fma.rn.f16x2 r3591, r2882, r3582, r3589;
}
{
mul.f16x2 r3595, r2882, r3584;
}
{
fma.rn.f16x2 r3598, r2894, r3582, r3595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3602, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3604, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3606, {low, high};
}
{
mul.f16x2 r3607, r3604, r3606;
}
{
mul.f16x2 r3610, r3578, r3602;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3613, {high, low};
}
{
fma.rn.f16x2 r3615, r3607, r3613, r3610;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3619, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3621, {high, high};
}
{
mul.f16x2 r3623, r3214, r3621;
}
{
neg.f16x2 r3626, r3623;
}
{
fma.rn.f16x2 r3628, r3202, r3619, r3626;
}
{
mul.f16x2 r3632, r3202, r3621;
}
{
fma.rn.f16x2 r3635, r3214, r3619, r3632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3639, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3641, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3643, {low, high};
}
{
mul.f16x2 r3644, r3641, r3643;
}
{
mul.f16x2 r3647, r3615, r3639;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3650, {high, low};
}
{
fma.rn.f16x2 r3652, r3644, r3650, r3647;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3656, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3658, {high, high};
}
{
mul.f16x2 r3660, r2114, r3658;
}
{
neg.f16x2 r3663, r3660;
}
{
fma.rn.f16x2 r3665, r1970, r3656, r3663;
}
{
mul.f16x2 r3669, r1970, r3658;
}
{
fma.rn.f16x2 r3672, r2114, r3656, r3669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3676, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3678, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3680, {low, high};
}
{
mul.f16x2 r3681, r3678, r3680;
}
{
mul.f16x2 r3684, r3652, r3676;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3687, {high, low};
}
{
fma.rn.f16x2 r3689, r3681, r3687, r3684;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3693, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3695, {high, high};
}
{
mul.f16x2 r3697, r2434, r3695;
}
{
neg.f16x2 r3700, r3697;
}
{
fma.rn.f16x2 r3702, r2290, r3693, r3700;
}
{
mul.f16x2 r3706, r2290, r3695;
}
{
fma.rn.f16x2 r3709, r2434, r3693, r3706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3713, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3715, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3717, {low, high};
}
{
mul.f16x2 r3718, r3715, r3717;
}
{
mul.f16x2 r3721, r3689, r3713;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3724, {high, low};
}
{
fma.rn.f16x2 r3726, r3718, r3724, r3721;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3730, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3732, {high, high};
}
{
mul.f16x2 r3734, r2754, r3732;
}
{
neg.f16x2 r3737, r3734;
}
{
fma.rn.f16x2 r3739, r2610, r3730, r3737;
}
{
mul.f16x2 r3743, r2610, r3732;
}
{
fma.rn.f16x2 r3746, r2754, r3730, r3743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3752, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3754, {low, high};
}
{
mul.f16x2 r3755, r3752, r3754;
}
{
mul.f16x2 r3758, r3726, r3750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3761, {high, low};
}
{
fma.rn.f16x2 r3763, r3755, r3761, r3758;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3767, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3769, {high, high};
}
{
mul.f16x2 r3771, r3074, r3769;
}
{
neg.f16x2 r3774, r3771;
}
{
fma.rn.f16x2 r3776, r2930, r3767, r3774;
}
{
mul.f16x2 r3780, r2930, r3769;
}
{
fma.rn.f16x2 r3783, r3074, r3767, r3780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3789, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3791, {low, high};
}
{
mul.f16x2 r3792, r3789, r3791;
}
{
mul.f16x2 r3795, r3763, r3787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3798, {high, low};
}
{
fma.rn.f16x2 r3800, r3792, r3798, r3795;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3804, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3806, {high, high};
}
{
mul.f16x2 r3808, r3394, r3806;
}
{
neg.f16x2 r3811, r3808;
}
{
fma.rn.f16x2 r3813, r3250, r3804, r3811;
}
{
mul.f16x2 r3817, r3250, r3806;
}
{
fma.rn.f16x2 r3820, r3394, r3804, r3817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3826, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3828, {low, high};
}
{
mul.f16x2 r3829, r3826, r3828;
}
{
mul.f16x2 r3832, r3800, r3824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3835, {high, low};
}
{
fma.rn.f16x2 r3837, r3829, r3835, r3832;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3841, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3843, {high, high};
}
{
mul.f16x2 r3845, r2186, r3843;
}
{
neg.f16x2 r3848, r3845;
}
{
fma.rn.f16x2 r3850, r2042, r3841, r3848;
}
{
mul.f16x2 r3854, r2042, r3843;
}
{
fma.rn.f16x2 r3857, r2186, r3841, r3854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3863, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3865, {low, high};
}
{
mul.f16x2 r3866, r3863, r3865;
}
{
mul.f16x2 r3869, r3837, r3861;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3872, {high, low};
}
{
fma.rn.f16x2 r3874, r3866, r3872, r3869;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3878, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3880, {high, high};
}
{
mul.f16x2 r3882, r2506, r3880;
}
{
neg.f16x2 r3885, r3882;
}
{
fma.rn.f16x2 r3887, r2362, r3878, r3885;
}
{
mul.f16x2 r3891, r2362, r3880;
}
{
fma.rn.f16x2 r3894, r2506, r3878, r3891;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3900, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3902, {low, high};
}
{
mul.f16x2 r3903, r3900, r3902;
}
{
mul.f16x2 r3906, r3874, r3898;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3909, {high, low};
}
{
fma.rn.f16x2 r3911, r3903, r3909, r3906;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3915, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3917, {high, high};
}
{
mul.f16x2 r3919, r2826, r3917;
}
{
neg.f16x2 r3922, r3919;
}
{
fma.rn.f16x2 r3924, r2682, r3915, r3922;
}
{
mul.f16x2 r3928, r2682, r3917;
}
{
fma.rn.f16x2 r3931, r2826, r3915, r3928;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3937, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3939, {low, high};
}
{
mul.f16x2 r3940, r3937, r3939;
}
{
mul.f16x2 r3943, r3911, r3935;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3946, {high, low};
}
{
fma.rn.f16x2 r3948, r3940, r3946, r3943;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3952, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3954, {high, high};
}
{
mul.f16x2 r3956, r3146, r3954;
}
{
neg.f16x2 r3959, r3956;
}
{
fma.rn.f16x2 r3961, r3002, r3952, r3959;
}
{
mul.f16x2 r3965, r3002, r3954;
}
{
fma.rn.f16x2 r3968, r3146, r3952, r3965;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3974, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3976, {low, high};
}
{
mul.f16x2 r3977, r3974, r3976;
}
{
mul.f16x2 r3980, r3948, r3972;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3983, {high, low};
}
{
fma.rn.f16x2 r3985, r3977, r3983, r3980;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r3989, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r3991, {high, high};
}
{
mul.f16x2 r3993, r3466, r3991;
}
{
neg.f16x2 r3996, r3993;
}
{
fma.rn.f16x2 r3998, r3322, r3989, r3996;
}
{
mul.f16x2 r4002, r3322, r3991;
}
{
fma.rn.f16x2 r4005, r3466, r3989, r4002;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4011, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4013, {low, high};
}
{
mul.f16x2 r4014, r4011, r4013;
}
{
mul.f16x2 r4017, r3985, r4009;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r4020, {high, low};
}
{
fma.rn.f16x2 r4022, r4014, r4020, r4017;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4026, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4028, {high, high};
}
{
mul.f16x2 r4030, r2222, r4028;
}
{
neg.f16x2 r4033, r4030;
}
{
fma.rn.f16x2 r4035, r2078, r4026, r4033;
}
{
mul.f16x2 r4039, r2078, r4028;
}
{
fma.rn.f16x2 r4042, r2222, r4026, r4039;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4048, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4050, {low, high};
}
{
mul.f16x2 r4051, r4048, r4050;
}
{
mul.f16x2 r4054, r4022, r4046;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4057, {high, low};
}
{
fma.rn.f16x2 r4059, r4051, r4057, r4054;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4063, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4065, {high, high};
}
{
mul.f16x2 r4067, r2542, r4065;
}
{
neg.f16x2 r4070, r4067;
}
{
fma.rn.f16x2 r4072, r2398, r4063, r4070;
}
{
mul.f16x2 r4076, r2398, r4065;
}
{
fma.rn.f16x2 r4079, r2542, r4063, r4076;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4085, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4087, {low, high};
}
{
mul.f16x2 r4088, r4085, r4087;
}
{
mul.f16x2 r4091, r4059, r4083;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4094, {high, low};
}
{
fma.rn.f16x2 r4096, r4088, r4094, r4091;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4100, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4102, {high, high};
}
{
mul.f16x2 r4104, r2862, r4102;
}
{
neg.f16x2 r4107, r4104;
}
{
fma.rn.f16x2 r4109, r2718, r4100, r4107;
}
{
mul.f16x2 r4113, r2718, r4102;
}
{
fma.rn.f16x2 r4116, r2862, r4100, r4113;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4122, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4124, {low, high};
}
{
mul.f16x2 r4125, r4122, r4124;
}
{
mul.f16x2 r4128, r4096, r4120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4131, {high, low};
}
{
fma.rn.f16x2 r4133, r4125, r4131, r4128;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4137, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4139, {high, high};
}
{
mul.f16x2 r4141, r3182, r4139;
}
{
neg.f16x2 r4144, r4141;
}
{
fma.rn.f16x2 r4146, r3038, r4137, r4144;
}
{
mul.f16x2 r4150, r3038, r4139;
}
{
fma.rn.f16x2 r4153, r3182, r4137, r4150;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4157, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4159, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4161, {low, high};
}
{
mul.f16x2 r4162, r4159, r4161;
}
{
mul.f16x2 r4165, r4133, r4157;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4168, {high, low};
}
{
fma.rn.f16x2 r4170, r4162, r4168, r4165;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4174, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4176, {high, high};
}
{
mul.f16x2 r4178, r3502, r4176;
}
{
neg.f16x2 r4181, r4178;
}
{
fma.rn.f16x2 r4183, r3358, r4174, r4181;
}
{
mul.f16x2 r4187, r3358, r4176;
}
{
fma.rn.f16x2 r4190, r3502, r4174, r4187;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4194, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4196, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4198, {low, high};
}
{
mul.f16x2 r4199, r4196, r4198;
}
{
mul.f16x2 r4202, r4170, r4194;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4205, {high, low};
}
{
fma.rn.f16x2 r4207, r4199, r4205, r4202;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4211, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4213, {high, high};
}
{
mul.f16x2 r4215, r2150, r4213;
}
{
neg.f16x2 r4218, r4215;
}
{
fma.rn.f16x2 r4220, r2006, r4211, r4218;
}
{
mul.f16x2 r4224, r2006, r4213;
}
{
fma.rn.f16x2 r4227, r2150, r4211, r4224;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4231, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4233, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4235, {low, high};
}
{
mul.f16x2 r4236, r4233, r4235;
}
{
mul.f16x2 r4239, r4207, r4231;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4242, {high, low};
}
{
fma.rn.f16x2 r4244, r4236, r4242, r4239;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4248, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4250, {high, high};
}
{
mul.f16x2 r4252, r2470, r4250;
}
{
neg.f16x2 r4255, r4252;
}
{
fma.rn.f16x2 r4257, r2326, r4248, r4255;
}
{
mul.f16x2 r4261, r2326, r4250;
}
{
fma.rn.f16x2 r4264, r2470, r4248, r4261;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4268, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4270, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4272, {low, high};
}
{
mul.f16x2 r4273, r4270, r4272;
}
{
mul.f16x2 r4276, r4244, r4268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4279, {high, low};
}
{
fma.rn.f16x2 r4281, r4273, r4279, r4276;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4285, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4287, {high, high};
}
{
mul.f16x2 r4289, r2790, r4287;
}
{
neg.f16x2 r4292, r4289;
}
{
fma.rn.f16x2 r4294, r2646, r4285, r4292;
}
{
mul.f16x2 r4298, r2646, r4287;
}
{
fma.rn.f16x2 r4301, r2790, r4285, r4298;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4305, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4307, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4309, {low, high};
}
{
mul.f16x2 r4310, r4307, r4309;
}
{
mul.f16x2 r4313, r4281, r4305;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4316, {high, low};
}
{
fma.rn.f16x2 r4318, r4310, r4316, r4313;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4322, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4324, {high, high};
}
{
mul.f16x2 r4326, r3110, r4324;
}
{
neg.f16x2 r4329, r4326;
}
{
fma.rn.f16x2 r4331, r2966, r4322, r4329;
}
{
mul.f16x2 r4335, r2966, r4324;
}
{
fma.rn.f16x2 r4338, r3110, r4322, r4335;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4342, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4344, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4346, {low, high};
}
{
mul.f16x2 r4347, r4344, r4346;
}
{
mul.f16x2 r4350, r4318, r4342;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4353, {high, low};
}
{
fma.rn.f16x2 r4355, r4347, r4353, r4350;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4355;
mov.b32 r4359, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4355;
mov.b32 r4361, {high, high};
}
{
mul.f16x2 r4363, r3430, r4361;
}
{
neg.f16x2 r4366, r4363;
}
{
fma.rn.f16x2 r4368, r3286, r4359, r4366;
}
{
mul.f16x2 r4372, r3286, r4361;
}
{
fma.rn.f16x2 r4375, r3430, r4359, r4372;
}
mad.lo.s32 r10398, r10395, 25000, r10393;
barrier.sync 0;
mad.lo.s32 r10399, r10397, 200, r10398;
st.shared.v2.f32 [r10399], {r1922, r1934};
st.shared.v2.f32 [r10399+8], {r3517, r3524};
st.shared.v2.f32 [r10399+16], {r3554, r3561};
st.shared.v2.f32 [r10399+24], {r3591, r3598};
st.shared.v2.f32 [r10399+32], {r3628, r3635};
st.shared.v2.f32 [r10399+40], {r3665, r3672};
st.shared.v2.f32 [r10399+48], {r3702, r3709};
st.shared.v2.f32 [r10399+56], {r3739, r3746};
st.shared.v2.f32 [r10399+64], {r3776, r3783};
st.shared.v2.f32 [r10399+72], {r3813, r3820};
st.shared.v2.f32 [r10399+80], {r3850, r3857};
st.shared.v2.f32 [r10399+88], {r3887, r3894};
st.shared.v2.f32 [r10399+96], {r3924, r3931};
st.shared.v2.f32 [r10399+104], {r3961, r3968};
st.shared.v2.f32 [r10399+112], {r3998, r4005};
st.shared.v2.f32 [r10399+120], {r4035, r4042};
st.shared.v2.f32 [r10399+128], {r4072, r4079};
st.shared.v2.f32 [r10399+136], {r4109, r4116};
st.shared.v2.f32 [r10399+144], {r4146, r4153};
st.shared.v2.f32 [r10399+152], {r4183, r4190};
st.shared.v2.f32 [r10399+160], {r4220, r4227};
st.shared.v2.f32 [r10399+168], {r4257, r4264};
st.shared.v2.f32 [r10399+176], {r4294, r4301};
st.shared.v2.f32 [r10399+184], {r4331, r4338};
st.shared.v2.f32 [r10399+192], {r4368, r4375};
barrier.sync 0;
mad.lo.s32 r10400, r10397, -192, r10399;
ld.shared.u32 r4408, [r10400];
ld.shared.u32 r4420, [r10400+4];
ld.shared.u32 r4728, [r10400+1000];
ld.shared.u32 r4740, [r10400+1004];
ld.shared.u32 r5048, [r10400+2000];
ld.shared.u32 r5060, [r10400+2004];
ld.shared.u32 r5368, [r10400+3000];
ld.shared.u32 r5380, [r10400+3004];
ld.shared.u32 r5688, [r10400+4000];
ld.shared.u32 r5700, [r10400+4004];
ld.shared.u32 r4405, [r10400+5000];
ld.shared.u32 r4417, [r10400+5004];
ld.shared.u32 r4725, [r10400+6000];
ld.shared.u32 r4737, [r10400+6004];
ld.shared.u32 r5045, [r10400+7000];
ld.shared.u32 r5057, [r10400+7004];
ld.shared.u32 r5365, [r10400+8000];
ld.shared.u32 r5377, [r10400+8004];
ld.shared.u32 r5685, [r10400+9000];
ld.shared.u32 r5697, [r10400+9004];
ld.shared.u32 r4411, [r10400+10000];
ld.shared.u32 r4423, [r10400+10004];
ld.shared.u32 r4731, [r10400+11000];
ld.shared.u32 r4743, [r10400+11004];
ld.shared.u32 r5051, [r10400+12000];
ld.shared.u32 r5063, [r10400+12004];
ld.shared.u32 r5371, [r10400+13000];
ld.shared.u32 r5383, [r10400+13004];
ld.shared.u32 r5691, [r10400+14000];
ld.shared.u32 r5703, [r10400+14004];
ld.shared.u32 r4412, [r10400+15000];
ld.shared.u32 r4424, [r10400+15004];
ld.shared.u32 r4732, [r10400+16000];
ld.shared.u32 r4744, [r10400+16004];
ld.shared.u32 r5052, [r10400+17000];
ld.shared.u32 r5064, [r10400+17004];
ld.shared.u32 r5372, [r10400+18000];
ld.shared.u32 r5384, [r10400+18004];
ld.shared.u32 r5692, [r10400+19000];
ld.shared.u32 r5704, [r10400+19004];
ld.shared.u32 r4406, [r10400+20000];
ld.shared.u32 r4418, [r10400+20004];
ld.shared.u32 r4726, [r10400+21000];
ld.shared.u32 r4738, [r10400+21004];
ld.shared.u32 r5046, [r10400+22000];
ld.shared.u32 r5058, [r10400+22004];
ld.shared.u32 r5366, [r10400+23000];
ld.shared.u32 r5378, [r10400+23004];
ld.shared.u32 r5686, [r10400+24000];
ld.shared.u32 r5698, [r10400+24004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4401, {low, high};
}
{
neg.f16x2 r4402, r4401;
}
{
add.f16x2 r4404, r4405, r4406;
}
{
add.f16x2 r4407, r4408, r4404;
}
{
add.f16x2 r4410, r4411, r4412;
}
{
add.f16x2 r4413, r4407, r4410;
}
{
add.f16x2 r4416, r4417, r4418;
}
{
add.f16x2 r4419, r4420, r4416;
}
{
add.f16x2 r4422, r4423, r4424;
}
{
add.f16x2 r4425, r4419, r4422;
}
{
add.f16x2 r4428, r4405, r4406;
}
{
mul.f16x2 r4431, r4428, r4396;
}
{
add.f16x2 r4434, r4408, r4431;
}
{
add.f16x2 r4437, r4411, r4412;
}
{
mul.f16x2 r4440, r4437, r4398;
}
{
add.f16x2 r4443, r4434, r4440;
}
{
sub.f16x2 r4446, r4417, r4418;
}
{
mul.f16x2 r4449, r4446, r4397;
}
{
sub.f16x2 r4452, r4423, r4424;
}
{
mul.f16x2 r4455, r4452, r4399;
}
{
add.f16x2 r4458, r4449, r4455;
}
{
sub.f16x2 r4461, r4443, r4458;
}
{
add.f16x2 r4464, r4405, r4406;
}
{
mul.f16x2 r4467, r4464, r4396;
}
{
add.f16x2 r4470, r4408, r4467;
}
{
add.f16x2 r4473, r4411, r4412;
}
{
mul.f16x2 r4476, r4473, r4398;
}
{
add.f16x2 r4479, r4470, r4476;
}
{
sub.f16x2 r4482, r4417, r4418;
}
{
mul.f16x2 r4485, r4482, r4397;
}
{
sub.f16x2 r4488, r4423, r4424;
}
{
mul.f16x2 r4491, r4488, r4399;
}
{
add.f16x2 r4494, r4485, r4491;
}
{
add.f16x2 r4497, r4479, r4494;
}
{
add.f16x2 r4500, r4405, r4406;
}
{
mul.f16x2 r4503, r4500, r4398;
}
{
add.f16x2 r4506, r4408, r4503;
}
{
add.f16x2 r4509, r4411, r4412;
}
{
mul.f16x2 r4512, r4509, r4400;
}
{
add.f16x2 r4515, r4506, r4512;
}
{
sub.f16x2 r4518, r4417, r4418;
}
{
mul.f16x2 r4521, r4518, r4399;
}
{
sub.f16x2 r4524, r4423, r4424;
}
{
mul.f16x2 r4527, r4524, r4402;
}
{
add.f16x2 r4530, r4521, r4527;
}
{
sub.f16x2 r4533, r4515, r4530;
}
{
add.f16x2 r4536, r4405, r4406;
}
{
mul.f16x2 r4539, r4536, r4398;
}
{
add.f16x2 r4542, r4408, r4539;
}
{
add.f16x2 r4545, r4411, r4412;
}
{
mul.f16x2 r4548, r4545, r4400;
}
{
add.f16x2 r4551, r4542, r4548;
}
{
sub.f16x2 r4554, r4417, r4418;
}
{
mul.f16x2 r4557, r4554, r4399;
}
{
sub.f16x2 r4560, r4423, r4424;
}
{
mul.f16x2 r4563, r4560, r4402;
}
{
add.f16x2 r4566, r4557, r4563;
}
{
add.f16x2 r4569, r4551, r4566;
}
{
add.f16x2 r4572, r4417, r4418;
}
{
mul.f16x2 r4575, r4572, r4396;
}
{
add.f16x2 r4578, r4420, r4575;
}
{
add.f16x2 r4581, r4423, r4424;
}
{
mul.f16x2 r4584, r4581, r4398;
}
{
add.f16x2 r4587, r4578, r4584;
}
{
sub.f16x2 r4590, r4405, r4406;
}
{
mul.f16x2 r4593, r4590, r4397;
}
{
sub.f16x2 r4596, r4411, r4412;
}
{
mul.f16x2 r4599, r4596, r4399;
}
{
add.f16x2 r4602, r4593, r4599;
}
{
add.f16x2 r4605, r4587, r4602;
}
{
add.f16x2 r4608, r4417, r4418;
}
{
mul.f16x2 r4611, r4608, r4396;
}
{
add.f16x2 r4614, r4420, r4611;
}
{
add.f16x2 r4617, r4423, r4424;
}
{
mul.f16x2 r4620, r4617, r4398;
}
{
add.f16x2 r4623, r4614, r4620;
}
{
sub.f16x2 r4626, r4405, r4406;
}
{
mul.f16x2 r4629, r4626, r4397;
}
{
sub.f16x2 r4632, r4411, r4412;
}
{
mul.f16x2 r4635, r4632, r4399;
}
{
add.f16x2 r4638, r4629, r4635;
}
{
sub.f16x2 r4641, r4623, r4638;
}
{
add.f16x2 r4644, r4417, r4418;
}
{
mul.f16x2 r4647, r4644, r4398;
}
{
add.f16x2 r4650, r4420, r4647;
}
{
add.f16x2 r4653, r4423, r4424;
}
{
mul.f16x2 r4656, r4653, r4400;
}
{
add.f16x2 r4659, r4650, r4656;
}
{
sub.f16x2 r4662, r4405, r4406;
}
{
mul.f16x2 r4665, r4662, r4399;
}
{
sub.f16x2 r4668, r4411, r4412;
}
{
mul.f16x2 r4671, r4668, r4402;
}
{
add.f16x2 r4674, r4665, r4671;
}
{
add.f16x2 r4677, r4659, r4674;
}
{
add.f16x2 r4680, r4417, r4418;
}
{
mul.f16x2 r4683, r4680, r4398;
}
{
add.f16x2 r4686, r4420, r4683;
}
{
add.f16x2 r4689, r4423, r4424;
}
{
mul.f16x2 r4692, r4689, r4400;
}
{
add.f16x2 r4695, r4686, r4692;
}
{
sub.f16x2 r4698, r4405, r4406;
}
{
mul.f16x2 r4701, r4698, r4399;
}
{
sub.f16x2 r4704, r4411, r4412;
}
{
mul.f16x2 r4707, r4704, r4402;
}
{
add.f16x2 r4710, r4701, r4707;
}
{
sub.f16x2 r4713, r4695, r4710;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4721, {low, high};
}
{
neg.f16x2 r4722, r4721;
}
{
add.f16x2 r4724, r4725, r4726;
}
{
add.f16x2 r4727, r4728, r4724;
}
{
add.f16x2 r4730, r4731, r4732;
}
{
add.f16x2 r4733, r4727, r4730;
}
{
add.f16x2 r4736, r4737, r4738;
}
{
add.f16x2 r4739, r4740, r4736;
}
{
add.f16x2 r4742, r4743, r4744;
}
{
add.f16x2 r4745, r4739, r4742;
}
{
add.f16x2 r4748, r4725, r4726;
}
{
mul.f16x2 r4751, r4748, r4716;
}
{
add.f16x2 r4754, r4728, r4751;
}
{
add.f16x2 r4757, r4731, r4732;
}
{
mul.f16x2 r4760, r4757, r4718;
}
{
add.f16x2 r4763, r4754, r4760;
}
{
sub.f16x2 r4766, r4737, r4738;
}
{
mul.f16x2 r4769, r4766, r4717;
}
{
sub.f16x2 r4772, r4743, r4744;
}
{
mul.f16x2 r4775, r4772, r4719;
}
{
add.f16x2 r4778, r4769, r4775;
}
{
sub.f16x2 r4781, r4763, r4778;
}
{
add.f16x2 r4784, r4725, r4726;
}
{
mul.f16x2 r4787, r4784, r4716;
}
{
add.f16x2 r4790, r4728, r4787;
}
{
add.f16x2 r4793, r4731, r4732;
}
{
mul.f16x2 r4796, r4793, r4718;
}
{
add.f16x2 r4799, r4790, r4796;
}
{
sub.f16x2 r4802, r4737, r4738;
}
{
mul.f16x2 r4805, r4802, r4717;
}
{
sub.f16x2 r4808, r4743, r4744;
}
{
mul.f16x2 r4811, r4808, r4719;
}
{
add.f16x2 r4814, r4805, r4811;
}
{
add.f16x2 r4817, r4799, r4814;
}
{
add.f16x2 r4820, r4725, r4726;
}
{
mul.f16x2 r4823, r4820, r4718;
}
{
add.f16x2 r4826, r4728, r4823;
}
{
add.f16x2 r4829, r4731, r4732;
}
{
mul.f16x2 r4832, r4829, r4720;
}
{
add.f16x2 r4835, r4826, r4832;
}
{
sub.f16x2 r4838, r4737, r4738;
}
{
mul.f16x2 r4841, r4838, r4719;
}
{
sub.f16x2 r4844, r4743, r4744;
}
{
mul.f16x2 r4847, r4844, r4722;
}
{
add.f16x2 r4850, r4841, r4847;
}
{
sub.f16x2 r4853, r4835, r4850;
}
{
add.f16x2 r4856, r4725, r4726;
}
{
mul.f16x2 r4859, r4856, r4718;
}
{
add.f16x2 r4862, r4728, r4859;
}
{
add.f16x2 r4865, r4731, r4732;
}
{
mul.f16x2 r4868, r4865, r4720;
}
{
add.f16x2 r4871, r4862, r4868;
}
{
sub.f16x2 r4874, r4737, r4738;
}
{
mul.f16x2 r4877, r4874, r4719;
}
{
sub.f16x2 r4880, r4743, r4744;
}
{
mul.f16x2 r4883, r4880, r4722;
}
{
add.f16x2 r4886, r4877, r4883;
}
{
add.f16x2 r4889, r4871, r4886;
}
{
add.f16x2 r4892, r4737, r4738;
}
{
mul.f16x2 r4895, r4892, r4716;
}
{
add.f16x2 r4898, r4740, r4895;
}
{
add.f16x2 r4901, r4743, r4744;
}
{
mul.f16x2 r4904, r4901, r4718;
}
{
add.f16x2 r4907, r4898, r4904;
}
{
sub.f16x2 r4910, r4725, r4726;
}
{
mul.f16x2 r4913, r4910, r4717;
}
{
sub.f16x2 r4916, r4731, r4732;
}
{
mul.f16x2 r4919, r4916, r4719;
}
{
add.f16x2 r4922, r4913, r4919;
}
{
add.f16x2 r4925, r4907, r4922;
}
{
add.f16x2 r4928, r4737, r4738;
}
{
mul.f16x2 r4931, r4928, r4716;
}
{
add.f16x2 r4934, r4740, r4931;
}
{
add.f16x2 r4937, r4743, r4744;
}
{
mul.f16x2 r4940, r4937, r4718;
}
{
add.f16x2 r4943, r4934, r4940;
}
{
sub.f16x2 r4946, r4725, r4726;
}
{
mul.f16x2 r4949, r4946, r4717;
}
{
sub.f16x2 r4952, r4731, r4732;
}
{
mul.f16x2 r4955, r4952, r4719;
}
{
add.f16x2 r4958, r4949, r4955;
}
{
sub.f16x2 r4961, r4943, r4958;
}
{
add.f16x2 r4964, r4737, r4738;
}
{
mul.f16x2 r4967, r4964, r4718;
}
{
add.f16x2 r4970, r4740, r4967;
}
{
add.f16x2 r4973, r4743, r4744;
}
{
mul.f16x2 r4976, r4973, r4720;
}
{
add.f16x2 r4979, r4970, r4976;
}
{
sub.f16x2 r4982, r4725, r4726;
}
{
mul.f16x2 r4985, r4982, r4719;
}
{
sub.f16x2 r4988, r4731, r4732;
}
{
mul.f16x2 r4991, r4988, r4722;
}
{
add.f16x2 r4994, r4985, r4991;
}
{
add.f16x2 r4997, r4979, r4994;
}
{
add.f16x2 r5000, r4737, r4738;
}
{
mul.f16x2 r5003, r5000, r4718;
}
{
add.f16x2 r5006, r4740, r5003;
}
{
add.f16x2 r5009, r4743, r4744;
}
{
mul.f16x2 r5012, r5009, r4720;
}
{
add.f16x2 r5015, r5006, r5012;
}
{
sub.f16x2 r5018, r4725, r4726;
}
{
mul.f16x2 r5021, r5018, r4719;
}
{
sub.f16x2 r5024, r4731, r4732;
}
{
mul.f16x2 r5027, r5024, r4722;
}
{
add.f16x2 r5030, r5021, r5027;
}
{
sub.f16x2 r5033, r5015, r5030;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5038, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5039, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5041, {low, high};
}
{
neg.f16x2 r5042, r5041;
}
{
add.f16x2 r5044, r5045, r5046;
}
{
add.f16x2 r5047, r5048, r5044;
}
{
add.f16x2 r5050, r5051, r5052;
}
{
add.f16x2 r5053, r5047, r5050;
}
{
add.f16x2 r5056, r5057, r5058;
}
{
add.f16x2 r5059, r5060, r5056;
}
{
add.f16x2 r5062, r5063, r5064;
}
{
add.f16x2 r5065, r5059, r5062;
}
{
add.f16x2 r5068, r5045, r5046;
}
{
mul.f16x2 r5071, r5068, r5036;
}
{
add.f16x2 r5074, r5048, r5071;
}
{
add.f16x2 r5077, r5051, r5052;
}
{
mul.f16x2 r5080, r5077, r5038;
}
{
add.f16x2 r5083, r5074, r5080;
}
{
sub.f16x2 r5086, r5057, r5058;
}
{
mul.f16x2 r5089, r5086, r5037;
}
{
sub.f16x2 r5092, r5063, r5064;
}
{
mul.f16x2 r5095, r5092, r5039;
}
{
add.f16x2 r5098, r5089, r5095;
}
{
sub.f16x2 r5101, r5083, r5098;
}
{
add.f16x2 r5104, r5045, r5046;
}
{
mul.f16x2 r5107, r5104, r5036;
}
{
add.f16x2 r5110, r5048, r5107;
}
{
add.f16x2 r5113, r5051, r5052;
}
{
mul.f16x2 r5116, r5113, r5038;
}
{
add.f16x2 r5119, r5110, r5116;
}
{
sub.f16x2 r5122, r5057, r5058;
}
{
mul.f16x2 r5125, r5122, r5037;
}
{
sub.f16x2 r5128, r5063, r5064;
}
{
mul.f16x2 r5131, r5128, r5039;
}
{
add.f16x2 r5134, r5125, r5131;
}
{
add.f16x2 r5137, r5119, r5134;
}
{
add.f16x2 r5140, r5045, r5046;
}
{
mul.f16x2 r5143, r5140, r5038;
}
{
add.f16x2 r5146, r5048, r5143;
}
{
add.f16x2 r5149, r5051, r5052;
}
{
mul.f16x2 r5152, r5149, r5040;
}
{
add.f16x2 r5155, r5146, r5152;
}
{
sub.f16x2 r5158, r5057, r5058;
}
{
mul.f16x2 r5161, r5158, r5039;
}
{
sub.f16x2 r5164, r5063, r5064;
}
{
mul.f16x2 r5167, r5164, r5042;
}
{
add.f16x2 r5170, r5161, r5167;
}
{
sub.f16x2 r5173, r5155, r5170;
}
{
add.f16x2 r5176, r5045, r5046;
}
{
mul.f16x2 r5179, r5176, r5038;
}
{
add.f16x2 r5182, r5048, r5179;
}
{
add.f16x2 r5185, r5051, r5052;
}
{
mul.f16x2 r5188, r5185, r5040;
}
{
add.f16x2 r5191, r5182, r5188;
}
{
sub.f16x2 r5194, r5057, r5058;
}
{
mul.f16x2 r5197, r5194, r5039;
}
{
sub.f16x2 r5200, r5063, r5064;
}
{
mul.f16x2 r5203, r5200, r5042;
}
{
add.f16x2 r5206, r5197, r5203;
}
{
add.f16x2 r5209, r5191, r5206;
}
{
add.f16x2 r5212, r5057, r5058;
}
{
mul.f16x2 r5215, r5212, r5036;
}
{
add.f16x2 r5218, r5060, r5215;
}
{
add.f16x2 r5221, r5063, r5064;
}
{
mul.f16x2 r5224, r5221, r5038;
}
{
add.f16x2 r5227, r5218, r5224;
}
{
sub.f16x2 r5230, r5045, r5046;
}
{
mul.f16x2 r5233, r5230, r5037;
}
{
sub.f16x2 r5236, r5051, r5052;
}
{
mul.f16x2 r5239, r5236, r5039;
}
{
add.f16x2 r5242, r5233, r5239;
}
{
add.f16x2 r5245, r5227, r5242;
}
{
add.f16x2 r5248, r5057, r5058;
}
{
mul.f16x2 r5251, r5248, r5036;
}
{
add.f16x2 r5254, r5060, r5251;
}
{
add.f16x2 r5257, r5063, r5064;
}
{
mul.f16x2 r5260, r5257, r5038;
}
{
add.f16x2 r5263, r5254, r5260;
}
{
sub.f16x2 r5266, r5045, r5046;
}
{
mul.f16x2 r5269, r5266, r5037;
}
{
sub.f16x2 r5272, r5051, r5052;
}
{
mul.f16x2 r5275, r5272, r5039;
}
{
add.f16x2 r5278, r5269, r5275;
}
{
sub.f16x2 r5281, r5263, r5278;
}
{
add.f16x2 r5284, r5057, r5058;
}
{
mul.f16x2 r5287, r5284, r5038;
}
{
add.f16x2 r5290, r5060, r5287;
}
{
add.f16x2 r5293, r5063, r5064;
}
{
mul.f16x2 r5296, r5293, r5040;
}
{
add.f16x2 r5299, r5290, r5296;
}
{
sub.f16x2 r5302, r5045, r5046;
}
{
mul.f16x2 r5305, r5302, r5039;
}
{
sub.f16x2 r5308, r5051, r5052;
}
{
mul.f16x2 r5311, r5308, r5042;
}
{
add.f16x2 r5314, r5305, r5311;
}
{
add.f16x2 r5317, r5299, r5314;
}
{
add.f16x2 r5320, r5057, r5058;
}
{
mul.f16x2 r5323, r5320, r5038;
}
{
add.f16x2 r5326, r5060, r5323;
}
{
add.f16x2 r5329, r5063, r5064;
}
{
mul.f16x2 r5332, r5329, r5040;
}
{
add.f16x2 r5335, r5326, r5332;
}
{
sub.f16x2 r5338, r5045, r5046;
}
{
mul.f16x2 r5341, r5338, r5039;
}
{
sub.f16x2 r5344, r5051, r5052;
}
{
mul.f16x2 r5347, r5344, r5042;
}
{
add.f16x2 r5350, r5341, r5347;
}
{
sub.f16x2 r5353, r5335, r5350;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5361, {low, high};
}
{
neg.f16x2 r5362, r5361;
}
{
add.f16x2 r5364, r5365, r5366;
}
{
add.f16x2 r5367, r5368, r5364;
}
{
add.f16x2 r5370, r5371, r5372;
}
{
add.f16x2 r5373, r5367, r5370;
}
{
add.f16x2 r5376, r5377, r5378;
}
{
add.f16x2 r5379, r5380, r5376;
}
{
add.f16x2 r5382, r5383, r5384;
}
{
add.f16x2 r5385, r5379, r5382;
}
{
add.f16x2 r5388, r5365, r5366;
}
{
mul.f16x2 r5391, r5388, r5356;
}
{
add.f16x2 r5394, r5368, r5391;
}
{
add.f16x2 r5397, r5371, r5372;
}
{
mul.f16x2 r5400, r5397, r5358;
}
{
add.f16x2 r5403, r5394, r5400;
}
{
sub.f16x2 r5406, r5377, r5378;
}
{
mul.f16x2 r5409, r5406, r5357;
}
{
sub.f16x2 r5412, r5383, r5384;
}
{
mul.f16x2 r5415, r5412, r5359;
}
{
add.f16x2 r5418, r5409, r5415;
}
{
sub.f16x2 r5421, r5403, r5418;
}
{
add.f16x2 r5424, r5365, r5366;
}
{
mul.f16x2 r5427, r5424, r5356;
}
{
add.f16x2 r5430, r5368, r5427;
}
{
add.f16x2 r5433, r5371, r5372;
}
{
mul.f16x2 r5436, r5433, r5358;
}
{
add.f16x2 r5439, r5430, r5436;
}
{
sub.f16x2 r5442, r5377, r5378;
}
{
mul.f16x2 r5445, r5442, r5357;
}
{
sub.f16x2 r5448, r5383, r5384;
}
{
mul.f16x2 r5451, r5448, r5359;
}
{
add.f16x2 r5454, r5445, r5451;
}
{
add.f16x2 r5457, r5439, r5454;
}
{
add.f16x2 r5460, r5365, r5366;
}
{
mul.f16x2 r5463, r5460, r5358;
}
{
add.f16x2 r5466, r5368, r5463;
}
{
add.f16x2 r5469, r5371, r5372;
}
{
mul.f16x2 r5472, r5469, r5360;
}
{
add.f16x2 r5475, r5466, r5472;
}
{
sub.f16x2 r5478, r5377, r5378;
}
{
mul.f16x2 r5481, r5478, r5359;
}
{
sub.f16x2 r5484, r5383, r5384;
}
{
mul.f16x2 r5487, r5484, r5362;
}
{
add.f16x2 r5490, r5481, r5487;
}
{
sub.f16x2 r5493, r5475, r5490;
}
{
add.f16x2 r5496, r5365, r5366;
}
{
mul.f16x2 r5499, r5496, r5358;
}
{
add.f16x2 r5502, r5368, r5499;
}
{
add.f16x2 r5505, r5371, r5372;
}
{
mul.f16x2 r5508, r5505, r5360;
}
{
add.f16x2 r5511, r5502, r5508;
}
{
sub.f16x2 r5514, r5377, r5378;
}
{
mul.f16x2 r5517, r5514, r5359;
}
{
sub.f16x2 r5520, r5383, r5384;
}
{
mul.f16x2 r5523, r5520, r5362;
}
{
add.f16x2 r5526, r5517, r5523;
}
{
add.f16x2 r5529, r5511, r5526;
}
{
add.f16x2 r5532, r5377, r5378;
}
{
mul.f16x2 r5535, r5532, r5356;
}
{
add.f16x2 r5538, r5380, r5535;
}
{
add.f16x2 r5541, r5383, r5384;
}
{
mul.f16x2 r5544, r5541, r5358;
}
{
add.f16x2 r5547, r5538, r5544;
}
{
sub.f16x2 r5550, r5365, r5366;
}
{
mul.f16x2 r5553, r5550, r5357;
}
{
sub.f16x2 r5556, r5371, r5372;
}
{
mul.f16x2 r5559, r5556, r5359;
}
{
add.f16x2 r5562, r5553, r5559;
}
{
add.f16x2 r5565, r5547, r5562;
}
{
add.f16x2 r5568, r5377, r5378;
}
{
mul.f16x2 r5571, r5568, r5356;
}
{
add.f16x2 r5574, r5380, r5571;
}
{
add.f16x2 r5577, r5383, r5384;
}
{
mul.f16x2 r5580, r5577, r5358;
}
{
add.f16x2 r5583, r5574, r5580;
}
{
sub.f16x2 r5586, r5365, r5366;
}
{
mul.f16x2 r5589, r5586, r5357;
}
{
sub.f16x2 r5592, r5371, r5372;
}
{
mul.f16x2 r5595, r5592, r5359;
}
{
add.f16x2 r5598, r5589, r5595;
}
{
sub.f16x2 r5601, r5583, r5598;
}
{
add.f16x2 r5604, r5377, r5378;
}
{
mul.f16x2 r5607, r5604, r5358;
}
{
add.f16x2 r5610, r5380, r5607;
}
{
add.f16x2 r5613, r5383, r5384;
}
{
mul.f16x2 r5616, r5613, r5360;
}
{
add.f16x2 r5619, r5610, r5616;
}
{
sub.f16x2 r5622, r5365, r5366;
}
{
mul.f16x2 r5625, r5622, r5359;
}
{
sub.f16x2 r5628, r5371, r5372;
}
{
mul.f16x2 r5631, r5628, r5362;
}
{
add.f16x2 r5634, r5625, r5631;
}
{
add.f16x2 r5637, r5619, r5634;
}
{
add.f16x2 r5640, r5377, r5378;
}
{
mul.f16x2 r5643, r5640, r5358;
}
{
add.f16x2 r5646, r5380, r5643;
}
{
add.f16x2 r5649, r5383, r5384;
}
{
mul.f16x2 r5652, r5649, r5360;
}
{
add.f16x2 r5655, r5646, r5652;
}
{
sub.f16x2 r5658, r5365, r5366;
}
{
mul.f16x2 r5661, r5658, r5359;
}
{
sub.f16x2 r5664, r5371, r5372;
}
{
mul.f16x2 r5667, r5664, r5362;
}
{
add.f16x2 r5670, r5661, r5667;
}
{
sub.f16x2 r5673, r5655, r5670;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5680, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5681, {low, high};
}
{
neg.f16x2 r5682, r5681;
}
{
add.f16x2 r5684, r5685, r5686;
}
{
add.f16x2 r5687, r5688, r5684;
}
{
add.f16x2 r5690, r5691, r5692;
}
{
add.f16x2 r5693, r5687, r5690;
}
{
add.f16x2 r5696, r5697, r5698;
}
{
add.f16x2 r5699, r5700, r5696;
}
{
add.f16x2 r5702, r5703, r5704;
}
{
add.f16x2 r5705, r5699, r5702;
}
{
add.f16x2 r5708, r5685, r5686;
}
{
mul.f16x2 r5711, r5708, r5676;
}
{
add.f16x2 r5714, r5688, r5711;
}
{
add.f16x2 r5717, r5691, r5692;
}
{
mul.f16x2 r5720, r5717, r5678;
}
{
add.f16x2 r5723, r5714, r5720;
}
{
sub.f16x2 r5726, r5697, r5698;
}
{
mul.f16x2 r5729, r5726, r5677;
}
{
sub.f16x2 r5732, r5703, r5704;
}
{
mul.f16x2 r5735, r5732, r5679;
}
{
add.f16x2 r5738, r5729, r5735;
}
{
sub.f16x2 r5741, r5723, r5738;
}
{
add.f16x2 r5744, r5685, r5686;
}
{
mul.f16x2 r5747, r5744, r5676;
}
{
add.f16x2 r5750, r5688, r5747;
}
{
add.f16x2 r5753, r5691, r5692;
}
{
mul.f16x2 r5756, r5753, r5678;
}
{
add.f16x2 r5759, r5750, r5756;
}
{
sub.f16x2 r5762, r5697, r5698;
}
{
mul.f16x2 r5765, r5762, r5677;
}
{
sub.f16x2 r5768, r5703, r5704;
}
{
mul.f16x2 r5771, r5768, r5679;
}
{
add.f16x2 r5774, r5765, r5771;
}
{
add.f16x2 r5777, r5759, r5774;
}
{
add.f16x2 r5780, r5685, r5686;
}
{
mul.f16x2 r5783, r5780, r5678;
}
{
add.f16x2 r5786, r5688, r5783;
}
{
add.f16x2 r5789, r5691, r5692;
}
{
mul.f16x2 r5792, r5789, r5680;
}
{
add.f16x2 r5795, r5786, r5792;
}
{
sub.f16x2 r5798, r5697, r5698;
}
{
mul.f16x2 r5801, r5798, r5679;
}
{
sub.f16x2 r5804, r5703, r5704;
}
{
mul.f16x2 r5807, r5804, r5682;
}
{
add.f16x2 r5810, r5801, r5807;
}
{
sub.f16x2 r5813, r5795, r5810;
}
{
add.f16x2 r5816, r5685, r5686;
}
{
mul.f16x2 r5819, r5816, r5678;
}
{
add.f16x2 r5822, r5688, r5819;
}
{
add.f16x2 r5825, r5691, r5692;
}
{
mul.f16x2 r5828, r5825, r5680;
}
{
add.f16x2 r5831, r5822, r5828;
}
{
sub.f16x2 r5834, r5697, r5698;
}
{
mul.f16x2 r5837, r5834, r5679;
}
{
sub.f16x2 r5840, r5703, r5704;
}
{
mul.f16x2 r5843, r5840, r5682;
}
{
add.f16x2 r5846, r5837, r5843;
}
{
add.f16x2 r5849, r5831, r5846;
}
{
add.f16x2 r5852, r5697, r5698;
}
{
mul.f16x2 r5855, r5852, r5676;
}
{
add.f16x2 r5858, r5700, r5855;
}
{
add.f16x2 r5861, r5703, r5704;
}
{
mul.f16x2 r5864, r5861, r5678;
}
{
add.f16x2 r5867, r5858, r5864;
}
{
sub.f16x2 r5870, r5685, r5686;
}
{
mul.f16x2 r5873, r5870, r5677;
}
{
sub.f16x2 r5876, r5691, r5692;
}
{
mul.f16x2 r5879, r5876, r5679;
}
{
add.f16x2 r5882, r5873, r5879;
}
{
add.f16x2 r5885, r5867, r5882;
}
{
add.f16x2 r5888, r5697, r5698;
}
{
mul.f16x2 r5891, r5888, r5676;
}
{
add.f16x2 r5894, r5700, r5891;
}
{
add.f16x2 r5897, r5703, r5704;
}
{
mul.f16x2 r5900, r5897, r5678;
}
{
add.f16x2 r5903, r5894, r5900;
}
{
sub.f16x2 r5906, r5685, r5686;
}
{
mul.f16x2 r5909, r5906, r5677;
}
{
sub.f16x2 r5912, r5691, r5692;
}
{
mul.f16x2 r5915, r5912, r5679;
}
{
add.f16x2 r5918, r5909, r5915;
}
{
sub.f16x2 r5921, r5903, r5918;
}
{
add.f16x2 r5924, r5697, r5698;
}
{
mul.f16x2 r5927, r5924, r5678;
}
{
add.f16x2 r5930, r5700, r5927;
}
{
add.f16x2 r5933, r5703, r5704;
}
{
mul.f16x2 r5936, r5933, r5680;
}
{
add.f16x2 r5939, r5930, r5936;
}
{
sub.f16x2 r5942, r5685, r5686;
}
{
mul.f16x2 r5945, r5942, r5679;
}
{
sub.f16x2 r5948, r5691, r5692;
}
{
mul.f16x2 r5951, r5948, r5682;
}
{
add.f16x2 r5954, r5945, r5951;
}
{
add.f16x2 r5957, r5939, r5954;
}
{
add.f16x2 r5960, r5697, r5698;
}
{
mul.f16x2 r5963, r5960, r5678;
}
{
add.f16x2 r5966, r5700, r5963;
}
{
add.f16x2 r5969, r5703, r5704;
}
{
mul.f16x2 r5972, r5969, r5680;
}
{
add.f16x2 r5975, r5966, r5972;
}
{
sub.f16x2 r5978, r5685, r5686;
}
{
mul.f16x2 r5981, r5978, r5679;
}
{
sub.f16x2 r5984, r5691, r5692;
}
{
mul.f16x2 r5987, r5984, r5682;
}
{
add.f16x2 r5990, r5981, r5987;
}
{
sub.f16x2 r5993, r5975, r5990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r5996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r5997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r5998, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r5999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r6000, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r6001, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r6002, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r6003, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r6006, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r6007, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r6010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r6011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r6013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r6018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r6019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6026, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r6027, {low, high};
}
{
mul.f16x2 r6044, r4781, r5996;
}
{
mul.f16x2 r6047, r4925, r5997;
}
{
sub.f16x2 r6050, r6044, r6047;
}
{
mul.f16x2 r6053, r4781, r5997;
}
{
fma.rn.f16x2 r6056, r4925, r5996, r6053;
}
{
mul.f16x2 r6060, r5101, r5998;
}
{
mul.f16x2 r6063, r5245, r5999;
}
{
sub.f16x2 r6066, r6060, r6063;
}
{
mul.f16x2 r6069, r5101, r5999;
}
{
fma.rn.f16x2 r6072, r5245, r5998, r6069;
}
{
mul.f16x2 r6076, r5421, r6000;
}
{
mul.f16x2 r6079, r5565, r6001;
}
{
sub.f16x2 r6082, r6076, r6079;
}
{
mul.f16x2 r6085, r5421, r6001;
}
{
fma.rn.f16x2 r6088, r5565, r6000, r6085;
}
{
mul.f16x2 r6092, r5741, r6002;
}
{
mul.f16x2 r6095, r5885, r6003;
}
{
sub.f16x2 r6098, r6092, r6095;
}
{
mul.f16x2 r6101, r5741, r6003;
}
{
fma.rn.f16x2 r6104, r5885, r6002, r6101;
}
{
mul.f16x2 r6108, r4853, r5998;
}
{
mul.f16x2 r6111, r4997, r5999;
}
{
sub.f16x2 r6114, r6108, r6111;
}
{
mul.f16x2 r6117, r4853, r5999;
}
{
fma.rn.f16x2 r6120, r4997, r5998, r6117;
}
{
mul.f16x2 r6124, r5173, r6002;
}
{
mul.f16x2 r6127, r5317, r6003;
}
{
sub.f16x2 r6130, r6124, r6127;
}
{
mul.f16x2 r6133, r5173, r6003;
}
{
fma.rn.f16x2 r6136, r5317, r6002, r6133;
}
{
mul.f16x2 r6140, r5493, r6006;
}
{
mul.f16x2 r6143, r5637, r6007;
}
{
sub.f16x2 r6146, r6140, r6143;
}
{
mul.f16x2 r6149, r5493, r6007;
}
{
fma.rn.f16x2 r6152, r5637, r6006, r6149;
}
{
mul.f16x2 r6156, r5813, r6010;
}
{
mul.f16x2 r6159, r5957, r6011;
}
{
sub.f16x2 r6162, r6156, r6159;
}
{
mul.f16x2 r6165, r5813, r6011;
}
{
fma.rn.f16x2 r6168, r5957, r6010, r6165;
}
{
mul.f16x2 r6172, r4889, r6000;
}
{
mul.f16x2 r6175, r5033, r6001;
}
{
sub.f16x2 r6178, r6172, r6175;
}
{
mul.f16x2 r6181, r4889, r6001;
}
{
fma.rn.f16x2 r6184, r5033, r6000, r6181;
}
{
mul.f16x2 r6188, r5209, r6006;
}
{
mul.f16x2 r6191, r5353, r6007;
}
{
sub.f16x2 r6194, r6188, r6191;
}
{
mul.f16x2 r6197, r5209, r6007;
}
{
fma.rn.f16x2 r6200, r5353, r6006, r6197;
}
{
mul.f16x2 r6204, r5529, r6012;
}
{
mul.f16x2 r6207, r5673, r6013;
}
{
sub.f16x2 r6210, r6204, r6207;
}
{
mul.f16x2 r6213, r5529, r6013;
}
{
fma.rn.f16x2 r6216, r5673, r6012, r6213;
}
{
mul.f16x2 r6220, r5849, r6018;
}
{
mul.f16x2 r6223, r5993, r6019;
}
{
sub.f16x2 r6226, r6220, r6223;
}
{
mul.f16x2 r6229, r5849, r6019;
}
{
fma.rn.f16x2 r6232, r5993, r6018, r6229;
}
{
mul.f16x2 r6236, r4817, r6002;
}
{
mul.f16x2 r6239, r4961, r6003;
}
{
sub.f16x2 r6242, r6236, r6239;
}
{
mul.f16x2 r6245, r4817, r6003;
}
{
fma.rn.f16x2 r6248, r4961, r6002, r6245;
}
{
mul.f16x2 r6252, r5137, r6010;
}
{
mul.f16x2 r6255, r5281, r6011;
}
{
sub.f16x2 r6258, r6252, r6255;
}
{
mul.f16x2 r6261, r5137, r6011;
}
{
fma.rn.f16x2 r6264, r5281, r6010, r6261;
}
{
mul.f16x2 r6268, r5457, r6018;
}
{
mul.f16x2 r6271, r5601, r6019;
}
{
sub.f16x2 r6274, r6268, r6271;
}
{
mul.f16x2 r6277, r5457, r6019;
}
{
fma.rn.f16x2 r6280, r5601, r6018, r6277;
}
{
mul.f16x2 r6284, r5777, r6026;
}
{
mul.f16x2 r6287, r5921, r6027;
}
{
sub.f16x2 r6290, r6284, r6287;
}
{
mul.f16x2 r6293, r5777, r6027;
}
{
fma.rn.f16x2 r6296, r5921, r6026, r6293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6302, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6305, {low, high};
}
{
neg.f16x2 r6306, r6305;
}
{
add.f16x2 r6308, r4733, r5693;
}
{
add.f16x2 r6311, r4413, r6308;
}
{
add.f16x2 r6314, r5053, r5373;
}
{
add.f16x2 r6317, r6311, r6314;
}
{
add.f16x2 r6320, r4745, r5705;
}
{
add.f16x2 r6323, r4425, r6320;
}
{
add.f16x2 r6326, r5065, r5385;
}
{
add.f16x2 r6329, r6323, r6326;
}
{
add.f16x2 r6332, r4733, r5693;
}
{
mul.f16x2 r6335, r6332, r6300;
}
{
add.f16x2 r6338, r4413, r6335;
}
{
add.f16x2 r6341, r5053, r5373;
}
{
mul.f16x2 r6344, r6341, r6302;
}
{
add.f16x2 r6347, r6338, r6344;
}
{
sub.f16x2 r6350, r4745, r5705;
}
{
mul.f16x2 r6353, r6350, r6301;
}
{
sub.f16x2 r6356, r5065, r5385;
}
{
mul.f16x2 r6359, r6356, r6303;
}
{
add.f16x2 r6362, r6353, r6359;
}
{
sub.f16x2 r6365, r6347, r6362;
}
{
add.f16x2 r6368, r4733, r5693;
}
{
mul.f16x2 r6371, r6368, r6300;
}
{
add.f16x2 r6374, r4413, r6371;
}
{
add.f16x2 r6377, r5053, r5373;
}
{
mul.f16x2 r6380, r6377, r6302;
}
{
add.f16x2 r6383, r6374, r6380;
}
{
sub.f16x2 r6386, r4745, r5705;
}
{
mul.f16x2 r6389, r6386, r6301;
}
{
sub.f16x2 r6392, r5065, r5385;
}
{
mul.f16x2 r6395, r6392, r6303;
}
{
add.f16x2 r6398, r6389, r6395;
}
{
add.f16x2 r6401, r6383, r6398;
}
{
add.f16x2 r6404, r4733, r5693;
}
{
mul.f16x2 r6407, r6404, r6302;
}
{
add.f16x2 r6410, r4413, r6407;
}
{
add.f16x2 r6413, r5053, r5373;
}
{
mul.f16x2 r6416, r6413, r6304;
}
{
add.f16x2 r6419, r6410, r6416;
}
{
sub.f16x2 r6422, r4745, r5705;
}
{
mul.f16x2 r6425, r6422, r6303;
}
{
sub.f16x2 r6428, r5065, r5385;
}
{
mul.f16x2 r6431, r6428, r6306;
}
{
add.f16x2 r6434, r6425, r6431;
}
{
sub.f16x2 r6437, r6419, r6434;
}
{
add.f16x2 r6440, r4733, r5693;
}
{
mul.f16x2 r6443, r6440, r6302;
}
{
add.f16x2 r6446, r4413, r6443;
}
{
add.f16x2 r6449, r5053, r5373;
}
{
mul.f16x2 r6452, r6449, r6304;
}
{
add.f16x2 r6455, r6446, r6452;
}
{
sub.f16x2 r6458, r4745, r5705;
}
{
mul.f16x2 r6461, r6458, r6303;
}
{
sub.f16x2 r6464, r5065, r5385;
}
{
mul.f16x2 r6467, r6464, r6306;
}
{
add.f16x2 r6470, r6461, r6467;
}
{
add.f16x2 r6473, r6455, r6470;
}
{
add.f16x2 r6476, r4745, r5705;
}
{
mul.f16x2 r6479, r6476, r6300;
}
{
add.f16x2 r6482, r4425, r6479;
}
{
add.f16x2 r6485, r5065, r5385;
}
{
mul.f16x2 r6488, r6485, r6302;
}
{
add.f16x2 r6491, r6482, r6488;
}
{
sub.f16x2 r6494, r4733, r5693;
}
{
mul.f16x2 r6497, r6494, r6301;
}
{
sub.f16x2 r6500, r5053, r5373;
}
{
mul.f16x2 r6503, r6500, r6303;
}
{
add.f16x2 r6506, r6497, r6503;
}
{
add.f16x2 r6509, r6491, r6506;
}
{
add.f16x2 r6512, r4745, r5705;
}
{
mul.f16x2 r6515, r6512, r6300;
}
{
add.f16x2 r6518, r4425, r6515;
}
{
add.f16x2 r6521, r5065, r5385;
}
{
mul.f16x2 r6524, r6521, r6302;
}
{
add.f16x2 r6527, r6518, r6524;
}
{
sub.f16x2 r6530, r4733, r5693;
}
{
mul.f16x2 r6533, r6530, r6301;
}
{
sub.f16x2 r6536, r5053, r5373;
}
{
mul.f16x2 r6539, r6536, r6303;
}
{
add.f16x2 r6542, r6533, r6539;
}
{
sub.f16x2 r6545, r6527, r6542;
}
{
add.f16x2 r6548, r4745, r5705;
}
{
mul.f16x2 r6551, r6548, r6302;
}
{
add.f16x2 r6554, r4425, r6551;
}
{
add.f16x2 r6557, r5065, r5385;
}
{
mul.f16x2 r6560, r6557, r6304;
}
{
add.f16x2 r6563, r6554, r6560;
}
{
sub.f16x2 r6566, r4733, r5693;
}
{
mul.f16x2 r6569, r6566, r6303;
}
{
sub.f16x2 r6572, r5053, r5373;
}
{
mul.f16x2 r6575, r6572, r6306;
}
{
add.f16x2 r6578, r6569, r6575;
}
{
add.f16x2 r6581, r6563, r6578;
}
{
add.f16x2 r6584, r4745, r5705;
}
{
mul.f16x2 r6587, r6584, r6302;
}
{
add.f16x2 r6590, r4425, r6587;
}
{
add.f16x2 r6593, r5065, r5385;
}
{
mul.f16x2 r6596, r6593, r6304;
}
{
add.f16x2 r6599, r6590, r6596;
}
{
sub.f16x2 r6602, r4733, r5693;
}
{
mul.f16x2 r6605, r6602, r6303;
}
{
sub.f16x2 r6608, r5053, r5373;
}
{
mul.f16x2 r6611, r6608, r6306;
}
{
add.f16x2 r6614, r6605, r6611;
}
{
sub.f16x2 r6617, r6599, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6623, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6625, {low, high};
}
{
neg.f16x2 r6626, r6625;
}
{
add.f16x2 r6628, r6050, r6098;
}
{
add.f16x2 r6631, r4461, r6628;
}
{
add.f16x2 r6634, r6066, r6082;
}
{
add.f16x2 r6637, r6631, r6634;
}
{
add.f16x2 r6640, r6056, r6104;
}
{
add.f16x2 r6643, r4605, r6640;
}
{
add.f16x2 r6646, r6072, r6088;
}
{
add.f16x2 r6649, r6643, r6646;
}
{
add.f16x2 r6652, r6050, r6098;
}
{
mul.f16x2 r6655, r6652, r6620;
}
{
add.f16x2 r6658, r4461, r6655;
}
{
add.f16x2 r6661, r6066, r6082;
}
{
mul.f16x2 r6664, r6661, r6622;
}
{
add.f16x2 r6667, r6658, r6664;
}
{
sub.f16x2 r6670, r6056, r6104;
}
{
mul.f16x2 r6673, r6670, r6621;
}
{
sub.f16x2 r6676, r6072, r6088;
}
{
mul.f16x2 r6679, r6676, r6623;
}
{
add.f16x2 r6682, r6673, r6679;
}
{
sub.f16x2 r6685, r6667, r6682;
}
{
add.f16x2 r6688, r6050, r6098;
}
{
mul.f16x2 r6691, r6688, r6620;
}
{
add.f16x2 r6694, r4461, r6691;
}
{
add.f16x2 r6697, r6066, r6082;
}
{
mul.f16x2 r6700, r6697, r6622;
}
{
add.f16x2 r6703, r6694, r6700;
}
{
sub.f16x2 r6706, r6056, r6104;
}
{
mul.f16x2 r6709, r6706, r6621;
}
{
sub.f16x2 r6712, r6072, r6088;
}
{
mul.f16x2 r6715, r6712, r6623;
}
{
add.f16x2 r6718, r6709, r6715;
}
{
add.f16x2 r6721, r6703, r6718;
}
{
add.f16x2 r6724, r6050, r6098;
}
{
mul.f16x2 r6727, r6724, r6622;
}
{
add.f16x2 r6730, r4461, r6727;
}
{
add.f16x2 r6733, r6066, r6082;
}
{
mul.f16x2 r6736, r6733, r6624;
}
{
add.f16x2 r6739, r6730, r6736;
}
{
sub.f16x2 r6742, r6056, r6104;
}
{
mul.f16x2 r6745, r6742, r6623;
}
{
sub.f16x2 r6748, r6072, r6088;
}
{
mul.f16x2 r6751, r6748, r6626;
}
{
add.f16x2 r6754, r6745, r6751;
}
{
sub.f16x2 r6757, r6739, r6754;
}
{
add.f16x2 r6760, r6050, r6098;
}
{
mul.f16x2 r6763, r6760, r6622;
}
{
add.f16x2 r6766, r4461, r6763;
}
{
add.f16x2 r6769, r6066, r6082;
}
{
mul.f16x2 r6772, r6769, r6624;
}
{
add.f16x2 r6775, r6766, r6772;
}
{
sub.f16x2 r6778, r6056, r6104;
}
{
mul.f16x2 r6781, r6778, r6623;
}
{
sub.f16x2 r6784, r6072, r6088;
}
{
mul.f16x2 r6787, r6784, r6626;
}
{
add.f16x2 r6790, r6781, r6787;
}
{
add.f16x2 r6793, r6775, r6790;
}
{
add.f16x2 r6796, r6056, r6104;
}
{
mul.f16x2 r6799, r6796, r6620;
}
{
add.f16x2 r6802, r4605, r6799;
}
{
add.f16x2 r6805, r6072, r6088;
}
{
mul.f16x2 r6808, r6805, r6622;
}
{
add.f16x2 r6811, r6802, r6808;
}
{
sub.f16x2 r6814, r6050, r6098;
}
{
mul.f16x2 r6817, r6814, r6621;
}
{
sub.f16x2 r6820, r6066, r6082;
}
{
mul.f16x2 r6823, r6820, r6623;
}
{
add.f16x2 r6826, r6817, r6823;
}
{
add.f16x2 r6829, r6811, r6826;
}
{
add.f16x2 r6832, r6056, r6104;
}
{
mul.f16x2 r6835, r6832, r6620;
}
{
add.f16x2 r6838, r4605, r6835;
}
{
add.f16x2 r6841, r6072, r6088;
}
{
mul.f16x2 r6844, r6841, r6622;
}
{
add.f16x2 r6847, r6838, r6844;
}
{
sub.f16x2 r6850, r6050, r6098;
}
{
mul.f16x2 r6853, r6850, r6621;
}
{
sub.f16x2 r6856, r6066, r6082;
}
{
mul.f16x2 r6859, r6856, r6623;
}
{
add.f16x2 r6862, r6853, r6859;
}
{
sub.f16x2 r6865, r6847, r6862;
}
{
add.f16x2 r6868, r6056, r6104;
}
{
mul.f16x2 r6871, r6868, r6622;
}
{
add.f16x2 r6874, r4605, r6871;
}
{
add.f16x2 r6877, r6072, r6088;
}
{
mul.f16x2 r6880, r6877, r6624;
}
{
add.f16x2 r6883, r6874, r6880;
}
{
sub.f16x2 r6886, r6050, r6098;
}
{
mul.f16x2 r6889, r6886, r6623;
}
{
sub.f16x2 r6892, r6066, r6082;
}
{
mul.f16x2 r6895, r6892, r6626;
}
{
add.f16x2 r6898, r6889, r6895;
}
{
add.f16x2 r6901, r6883, r6898;
}
{
add.f16x2 r6904, r6056, r6104;
}
{
mul.f16x2 r6907, r6904, r6622;
}
{
add.f16x2 r6910, r4605, r6907;
}
{
add.f16x2 r6913, r6072, r6088;
}
{
mul.f16x2 r6916, r6913, r6624;
}
{
add.f16x2 r6919, r6910, r6916;
}
{
sub.f16x2 r6922, r6050, r6098;
}
{
mul.f16x2 r6925, r6922, r6623;
}
{
sub.f16x2 r6928, r6066, r6082;
}
{
mul.f16x2 r6931, r6928, r6626;
}
{
add.f16x2 r6934, r6925, r6931;
}
{
sub.f16x2 r6937, r6919, r6934;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6945, {low, high};
}
{
neg.f16x2 r6946, r6945;
}
{
add.f16x2 r6948, r6114, r6162;
}
{
add.f16x2 r6951, r4533, r6948;
}
{
add.f16x2 r6954, r6130, r6146;
}
{
add.f16x2 r6957, r6951, r6954;
}
{
add.f16x2 r6960, r6120, r6168;
}
{
add.f16x2 r6963, r4677, r6960;
}
{
add.f16x2 r6966, r6136, r6152;
}
{
add.f16x2 r6969, r6963, r6966;
}
{
add.f16x2 r6972, r6114, r6162;
}
{
mul.f16x2 r6975, r6972, r6940;
}
{
add.f16x2 r6978, r4533, r6975;
}
{
add.f16x2 r6981, r6130, r6146;
}
{
mul.f16x2 r6984, r6981, r6942;
}
{
add.f16x2 r6987, r6978, r6984;
}
{
sub.f16x2 r6990, r6120, r6168;
}
{
mul.f16x2 r6993, r6990, r6941;
}
{
sub.f16x2 r6996, r6136, r6152;
}
{
mul.f16x2 r6999, r6996, r6943;
}
{
add.f16x2 r7002, r6993, r6999;
}
{
sub.f16x2 r7005, r6987, r7002;
}
{
add.f16x2 r7008, r6114, r6162;
}
{
mul.f16x2 r7011, r7008, r6940;
}
{
add.f16x2 r7014, r4533, r7011;
}
{
add.f16x2 r7017, r6130, r6146;
}
{
mul.f16x2 r7020, r7017, r6942;
}
{
add.f16x2 r7023, r7014, r7020;
}
{
sub.f16x2 r7026, r6120, r6168;
}
{
mul.f16x2 r7029, r7026, r6941;
}
{
sub.f16x2 r7032, r6136, r6152;
}
{
mul.f16x2 r7035, r7032, r6943;
}
{
add.f16x2 r7038, r7029, r7035;
}
{
add.f16x2 r7041, r7023, r7038;
}
{
add.f16x2 r7044, r6114, r6162;
}
{
mul.f16x2 r7047, r7044, r6942;
}
{
add.f16x2 r7050, r4533, r7047;
}
{
add.f16x2 r7053, r6130, r6146;
}
{
mul.f16x2 r7056, r7053, r6944;
}
{
add.f16x2 r7059, r7050, r7056;
}
{
sub.f16x2 r7062, r6120, r6168;
}
{
mul.f16x2 r7065, r7062, r6943;
}
{
sub.f16x2 r7068, r6136, r6152;
}
{
mul.f16x2 r7071, r7068, r6946;
}
{
add.f16x2 r7074, r7065, r7071;
}
{
sub.f16x2 r7077, r7059, r7074;
}
{
add.f16x2 r7080, r6114, r6162;
}
{
mul.f16x2 r7083, r7080, r6942;
}
{
add.f16x2 r7086, r4533, r7083;
}
{
add.f16x2 r7089, r6130, r6146;
}
{
mul.f16x2 r7092, r7089, r6944;
}
{
add.f16x2 r7095, r7086, r7092;
}
{
sub.f16x2 r7098, r6120, r6168;
}
{
mul.f16x2 r7101, r7098, r6943;
}
{
sub.f16x2 r7104, r6136, r6152;
}
{
mul.f16x2 r7107, r7104, r6946;
}
{
add.f16x2 r7110, r7101, r7107;
}
{
add.f16x2 r7113, r7095, r7110;
}
{
add.f16x2 r7116, r6120, r6168;
}
{
mul.f16x2 r7119, r7116, r6940;
}
{
add.f16x2 r7122, r4677, r7119;
}
{
add.f16x2 r7125, r6136, r6152;
}
{
mul.f16x2 r7128, r7125, r6942;
}
{
add.f16x2 r7131, r7122, r7128;
}
{
sub.f16x2 r7134, r6114, r6162;
}
{
mul.f16x2 r7137, r7134, r6941;
}
{
sub.f16x2 r7140, r6130, r6146;
}
{
mul.f16x2 r7143, r7140, r6943;
}
{
add.f16x2 r7146, r7137, r7143;
}
{
add.f16x2 r7149, r7131, r7146;
}
{
add.f16x2 r7152, r6120, r6168;
}
{
mul.f16x2 r7155, r7152, r6940;
}
{
add.f16x2 r7158, r4677, r7155;
}
{
add.f16x2 r7161, r6136, r6152;
}
{
mul.f16x2 r7164, r7161, r6942;
}
{
add.f16x2 r7167, r7158, r7164;
}
{
sub.f16x2 r7170, r6114, r6162;
}
{
mul.f16x2 r7173, r7170, r6941;
}
{
sub.f16x2 r7176, r6130, r6146;
}
{
mul.f16x2 r7179, r7176, r6943;
}
{
add.f16x2 r7182, r7173, r7179;
}
{
sub.f16x2 r7185, r7167, r7182;
}
{
add.f16x2 r7188, r6120, r6168;
}
{
mul.f16x2 r7191, r7188, r6942;
}
{
add.f16x2 r7194, r4677, r7191;
}
{
add.f16x2 r7197, r6136, r6152;
}
{
mul.f16x2 r7200, r7197, r6944;
}
{
add.f16x2 r7203, r7194, r7200;
}
{
sub.f16x2 r7206, r6114, r6162;
}
{
mul.f16x2 r7209, r7206, r6943;
}
{
sub.f16x2 r7212, r6130, r6146;
}
{
mul.f16x2 r7215, r7212, r6946;
}
{
add.f16x2 r7218, r7209, r7215;
}
{
add.f16x2 r7221, r7203, r7218;
}
{
add.f16x2 r7224, r6120, r6168;
}
{
mul.f16x2 r7227, r7224, r6942;
}
{
add.f16x2 r7230, r4677, r7227;
}
{
add.f16x2 r7233, r6136, r6152;
}
{
mul.f16x2 r7236, r7233, r6944;
}
{
add.f16x2 r7239, r7230, r7236;
}
{
sub.f16x2 r7242, r6114, r6162;
}
{
mul.f16x2 r7245, r7242, r6943;
}
{
sub.f16x2 r7248, r6130, r6146;
}
{
mul.f16x2 r7251, r7248, r6946;
}
{
add.f16x2 r7254, r7245, r7251;
}
{
sub.f16x2 r7257, r7239, r7254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7265, {low, high};
}
{
neg.f16x2 r7266, r7265;
}
{
add.f16x2 r7268, r6178, r6226;
}
{
add.f16x2 r7271, r4569, r7268;
}
{
add.f16x2 r7274, r6194, r6210;
}
{
add.f16x2 r7277, r7271, r7274;
}
{
add.f16x2 r7280, r6184, r6232;
}
{
add.f16x2 r7283, r4713, r7280;
}
{
add.f16x2 r7286, r6200, r6216;
}
{
add.f16x2 r7289, r7283, r7286;
}
{
add.f16x2 r7292, r6178, r6226;
}
{
mul.f16x2 r7295, r7292, r7260;
}
{
add.f16x2 r7298, r4569, r7295;
}
{
add.f16x2 r7301, r6194, r6210;
}
{
mul.f16x2 r7304, r7301, r7262;
}
{
add.f16x2 r7307, r7298, r7304;
}
{
sub.f16x2 r7310, r6184, r6232;
}
{
mul.f16x2 r7313, r7310, r7261;
}
{
sub.f16x2 r7316, r6200, r6216;
}
{
mul.f16x2 r7319, r7316, r7263;
}
{
add.f16x2 r7322, r7313, r7319;
}
{
sub.f16x2 r7325, r7307, r7322;
}
{
add.f16x2 r7328, r6178, r6226;
}
{
mul.f16x2 r7331, r7328, r7260;
}
{
add.f16x2 r7334, r4569, r7331;
}
{
add.f16x2 r7337, r6194, r6210;
}
{
mul.f16x2 r7340, r7337, r7262;
}
{
add.f16x2 r7343, r7334, r7340;
}
{
sub.f16x2 r7346, r6184, r6232;
}
{
mul.f16x2 r7349, r7346, r7261;
}
{
sub.f16x2 r7352, r6200, r6216;
}
{
mul.f16x2 r7355, r7352, r7263;
}
{
add.f16x2 r7358, r7349, r7355;
}
{
add.f16x2 r7361, r7343, r7358;
}
{
add.f16x2 r7364, r6178, r6226;
}
{
mul.f16x2 r7367, r7364, r7262;
}
{
add.f16x2 r7370, r4569, r7367;
}
{
add.f16x2 r7373, r6194, r6210;
}
{
mul.f16x2 r7376, r7373, r7264;
}
{
add.f16x2 r7379, r7370, r7376;
}
{
sub.f16x2 r7382, r6184, r6232;
}
{
mul.f16x2 r7385, r7382, r7263;
}
{
sub.f16x2 r7388, r6200, r6216;
}
{
mul.f16x2 r7391, r7388, r7266;
}
{
add.f16x2 r7394, r7385, r7391;
}
{
sub.f16x2 r7397, r7379, r7394;
}
{
add.f16x2 r7400, r6178, r6226;
}
{
mul.f16x2 r7403, r7400, r7262;
}
{
add.f16x2 r7406, r4569, r7403;
}
{
add.f16x2 r7409, r6194, r6210;
}
{
mul.f16x2 r7412, r7409, r7264;
}
{
add.f16x2 r7415, r7406, r7412;
}
{
sub.f16x2 r7418, r6184, r6232;
}
{
mul.f16x2 r7421, r7418, r7263;
}
{
sub.f16x2 r7424, r6200, r6216;
}
{
mul.f16x2 r7427, r7424, r7266;
}
{
add.f16x2 r7430, r7421, r7427;
}
{
add.f16x2 r7433, r7415, r7430;
}
{
add.f16x2 r7436, r6184, r6232;
}
{
mul.f16x2 r7439, r7436, r7260;
}
{
add.f16x2 r7442, r4713, r7439;
}
{
add.f16x2 r7445, r6200, r6216;
}
{
mul.f16x2 r7448, r7445, r7262;
}
{
add.f16x2 r7451, r7442, r7448;
}
{
sub.f16x2 r7454, r6178, r6226;
}
{
mul.f16x2 r7457, r7454, r7261;
}
{
sub.f16x2 r7460, r6194, r6210;
}
{
mul.f16x2 r7463, r7460, r7263;
}
{
add.f16x2 r7466, r7457, r7463;
}
{
add.f16x2 r7469, r7451, r7466;
}
{
add.f16x2 r7472, r6184, r6232;
}
{
mul.f16x2 r7475, r7472, r7260;
}
{
add.f16x2 r7478, r4713, r7475;
}
{
add.f16x2 r7481, r6200, r6216;
}
{
mul.f16x2 r7484, r7481, r7262;
}
{
add.f16x2 r7487, r7478, r7484;
}
{
sub.f16x2 r7490, r6178, r6226;
}
{
mul.f16x2 r7493, r7490, r7261;
}
{
sub.f16x2 r7496, r6194, r6210;
}
{
mul.f16x2 r7499, r7496, r7263;
}
{
add.f16x2 r7502, r7493, r7499;
}
{
sub.f16x2 r7505, r7487, r7502;
}
{
add.f16x2 r7508, r6184, r6232;
}
{
mul.f16x2 r7511, r7508, r7262;
}
{
add.f16x2 r7514, r4713, r7511;
}
{
add.f16x2 r7517, r6200, r6216;
}
{
mul.f16x2 r7520, r7517, r7264;
}
{
add.f16x2 r7523, r7514, r7520;
}
{
sub.f16x2 r7526, r6178, r6226;
}
{
mul.f16x2 r7529, r7526, r7263;
}
{
sub.f16x2 r7532, r6194, r6210;
}
{
mul.f16x2 r7535, r7532, r7266;
}
{
add.f16x2 r7538, r7529, r7535;
}
{
add.f16x2 r7541, r7523, r7538;
}
{
add.f16x2 r7544, r6184, r6232;
}
{
mul.f16x2 r7547, r7544, r7262;
}
{
add.f16x2 r7550, r4713, r7547;
}
{
add.f16x2 r7553, r6200, r6216;
}
{
mul.f16x2 r7556, r7553, r7264;
}
{
add.f16x2 r7559, r7550, r7556;
}
{
sub.f16x2 r7562, r6178, r6226;
}
{
mul.f16x2 r7565, r7562, r7263;
}
{
sub.f16x2 r7568, r6194, r6210;
}
{
mul.f16x2 r7571, r7568, r7266;
}
{
add.f16x2 r7574, r7565, r7571;
}
{
sub.f16x2 r7577, r7559, r7574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7582, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7584, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7585, {low, high};
}
{
neg.f16x2 r7586, r7585;
}
{
add.f16x2 r7588, r6242, r6290;
}
{
add.f16x2 r7591, r4497, r7588;
}
{
add.f16x2 r7594, r6258, r6274;
}
{
add.f16x2 r7597, r7591, r7594;
}
{
add.f16x2 r7600, r6248, r6296;
}
{
add.f16x2 r7603, r4641, r7600;
}
{
add.f16x2 r7606, r6264, r6280;
}
{
add.f16x2 r7609, r7603, r7606;
}
{
add.f16x2 r7612, r6242, r6290;
}
{
mul.f16x2 r7615, r7612, r7580;
}
{
add.f16x2 r7618, r4497, r7615;
}
{
add.f16x2 r7621, r6258, r6274;
}
{
mul.f16x2 r7624, r7621, r7582;
}
{
add.f16x2 r7627, r7618, r7624;
}
{
sub.f16x2 r7630, r6248, r6296;
}
{
mul.f16x2 r7633, r7630, r7581;
}
{
sub.f16x2 r7636, r6264, r6280;
}
{
mul.f16x2 r7639, r7636, r7583;
}
{
add.f16x2 r7642, r7633, r7639;
}
{
sub.f16x2 r7645, r7627, r7642;
}
{
add.f16x2 r7648, r6242, r6290;
}
{
mul.f16x2 r7651, r7648, r7580;
}
{
add.f16x2 r7654, r4497, r7651;
}
{
add.f16x2 r7657, r6258, r6274;
}
{
mul.f16x2 r7660, r7657, r7582;
}
{
add.f16x2 r7663, r7654, r7660;
}
{
sub.f16x2 r7666, r6248, r6296;
}
{
mul.f16x2 r7669, r7666, r7581;
}
{
sub.f16x2 r7672, r6264, r6280;
}
{
mul.f16x2 r7675, r7672, r7583;
}
{
add.f16x2 r7678, r7669, r7675;
}
{
add.f16x2 r7681, r7663, r7678;
}
{
add.f16x2 r7684, r6242, r6290;
}
{
mul.f16x2 r7687, r7684, r7582;
}
{
add.f16x2 r7690, r4497, r7687;
}
{
add.f16x2 r7693, r6258, r6274;
}
{
mul.f16x2 r7696, r7693, r7584;
}
{
add.f16x2 r7699, r7690, r7696;
}
{
sub.f16x2 r7702, r6248, r6296;
}
{
mul.f16x2 r7705, r7702, r7583;
}
{
sub.f16x2 r7708, r6264, r6280;
}
{
mul.f16x2 r7711, r7708, r7586;
}
{
add.f16x2 r7714, r7705, r7711;
}
{
sub.f16x2 r7717, r7699, r7714;
}
{
add.f16x2 r7720, r6242, r6290;
}
{
mul.f16x2 r7723, r7720, r7582;
}
{
add.f16x2 r7726, r4497, r7723;
}
{
add.f16x2 r7729, r6258, r6274;
}
{
mul.f16x2 r7732, r7729, r7584;
}
{
add.f16x2 r7735, r7726, r7732;
}
{
sub.f16x2 r7738, r6248, r6296;
}
{
mul.f16x2 r7741, r7738, r7583;
}
{
sub.f16x2 r7744, r6264, r6280;
}
{
mul.f16x2 r7747, r7744, r7586;
}
{
add.f16x2 r7750, r7741, r7747;
}
{
add.f16x2 r7753, r7735, r7750;
}
{
add.f16x2 r7756, r6248, r6296;
}
{
mul.f16x2 r7759, r7756, r7580;
}
{
add.f16x2 r7762, r4641, r7759;
}
{
add.f16x2 r7765, r6264, r6280;
}
{
mul.f16x2 r7768, r7765, r7582;
}
{
add.f16x2 r7771, r7762, r7768;
}
{
sub.f16x2 r7774, r6242, r6290;
}
{
mul.f16x2 r7777, r7774, r7581;
}
{
sub.f16x2 r7780, r6258, r6274;
}
{
mul.f16x2 r7783, r7780, r7583;
}
{
add.f16x2 r7786, r7777, r7783;
}
{
add.f16x2 r7789, r7771, r7786;
}
{
add.f16x2 r7792, r6248, r6296;
}
{
mul.f16x2 r7795, r7792, r7580;
}
{
add.f16x2 r7798, r4641, r7795;
}
{
add.f16x2 r7801, r6264, r6280;
}
{
mul.f16x2 r7804, r7801, r7582;
}
{
add.f16x2 r7807, r7798, r7804;
}
{
sub.f16x2 r7810, r6242, r6290;
}
{
mul.f16x2 r7813, r7810, r7581;
}
{
sub.f16x2 r7816, r6258, r6274;
}
{
mul.f16x2 r7819, r7816, r7583;
}
{
add.f16x2 r7822, r7813, r7819;
}
{
sub.f16x2 r7825, r7807, r7822;
}
{
add.f16x2 r7828, r6248, r6296;
}
{
mul.f16x2 r7831, r7828, r7582;
}
{
add.f16x2 r7834, r4641, r7831;
}
{
add.f16x2 r7837, r6264, r6280;
}
{
mul.f16x2 r7840, r7837, r7584;
}
{
add.f16x2 r7843, r7834, r7840;
}
{
sub.f16x2 r7846, r6242, r6290;
}
{
mul.f16x2 r7849, r7846, r7583;
}
{
sub.f16x2 r7852, r6258, r6274;
}
{
mul.f16x2 r7855, r7852, r7586;
}
{
add.f16x2 r7858, r7849, r7855;
}
{
add.f16x2 r7861, r7843, r7858;
}
{
add.f16x2 r7864, r6248, r6296;
}
{
mul.f16x2 r7867, r7864, r7582;
}
{
add.f16x2 r7870, r4641, r7867;
}
{
add.f16x2 r7873, r6264, r6280;
}
{
mul.f16x2 r7876, r7873, r7584;
}
{
add.f16x2 r7879, r7870, r7876;
}
{
sub.f16x2 r7882, r6242, r6290;
}
{
mul.f16x2 r7885, r7882, r7583;
}
{
sub.f16x2 r7888, r6258, r6274;
}
{
mul.f16x2 r7891, r7888, r7586;
}
{
add.f16x2 r7894, r7885, r7891;
}
{
sub.f16x2 r7897, r7879, r7894;
}
mul.wide.u32 rd4, r10397, 1374389535;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r10401, rd5;
cvt.rn.f32.u32 f600, r10401;
mul.f32 f601, f600, 0f3D4DE32E;
cos.approx.f32 f485, f601;
sin.approx.f32 f602, f601;
neg.f32 f486, f602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f485;
cvt.rn.f16.f32 high, f486;
mov.b32 r7900, {low, high};
}
mul.lo.s32 r10402, r10401, 25;
sub.s32 r10403, r10397, r10402;
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7903, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7905, {high, high};
}
{
mul.f16x2 r7907, r6649, r7905;
}
{
neg.f16x2 r7910, r7907;
}
{
fma.rn.f16x2 r7912, r6637, r7903, r7910;
}
{
mul.f16x2 r7916, r6637, r7905;
}
{
fma.rn.f16x2 r7919, r6649, r7903, r7916;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7923, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7925, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7927, {low, high};
}
{
mul.f16x2 r7928, r7925, r7927;
}
{
mul.f16x2 r7931, r7900, r7923;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7934, {high, low};
}
{
fma.rn.f16x2 r7936, r7928, r7934, r7931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7940, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7942, {high, high};
}
{
mul.f16x2 r7944, r6969, r7942;
}
{
neg.f16x2 r7947, r7944;
}
{
fma.rn.f16x2 r7949, r6957, r7940, r7947;
}
{
mul.f16x2 r7953, r6957, r7942;
}
{
fma.rn.f16x2 r7956, r6969, r7940, r7953;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7960, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7962, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7964, {low, high};
}
{
mul.f16x2 r7965, r7962, r7964;
}
{
mul.f16x2 r7968, r7936, r7960;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7971, {high, low};
}
{
fma.rn.f16x2 r7973, r7965, r7971, r7968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r7977, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r7979, {high, high};
}
{
mul.f16x2 r7981, r7289, r7979;
}
{
neg.f16x2 r7984, r7981;
}
{
fma.rn.f16x2 r7986, r7277, r7977, r7984;
}
{
mul.f16x2 r7990, r7277, r7979;
}
{
fma.rn.f16x2 r7993, r7289, r7977, r7990;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7997, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7999, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8001, {low, high};
}
{
mul.f16x2 r8002, r7999, r8001;
}
{
mul.f16x2 r8005, r7973, r7997;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r8008, {high, low};
}
{
fma.rn.f16x2 r8010, r8002, r8008, r8005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8014, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8016, {high, high};
}
{
mul.f16x2 r8018, r7609, r8016;
}
{
neg.f16x2 r8021, r8018;
}
{
fma.rn.f16x2 r8023, r7597, r8014, r8021;
}
{
mul.f16x2 r8027, r7597, r8016;
}
{
fma.rn.f16x2 r8030, r7609, r8014, r8027;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8034, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8036, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8038, {low, high};
}
{
mul.f16x2 r8039, r8036, r8038;
}
{
mul.f16x2 r8042, r8010, r8034;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8045, {high, low};
}
{
fma.rn.f16x2 r8047, r8039, r8045, r8042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8051, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8053, {high, high};
}
{
mul.f16x2 r8055, r6509, r8053;
}
{
neg.f16x2 r8058, r8055;
}
{
fma.rn.f16x2 r8060, r6365, r8051, r8058;
}
{
mul.f16x2 r8064, r6365, r8053;
}
{
fma.rn.f16x2 r8067, r6509, r8051, r8064;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8071, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8073, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8075, {low, high};
}
{
mul.f16x2 r8076, r8073, r8075;
}
{
mul.f16x2 r8079, r8047, r8071;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8082, {high, low};
}
{
fma.rn.f16x2 r8084, r8076, r8082, r8079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8088, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8090, {high, high};
}
{
mul.f16x2 r8092, r6829, r8090;
}
{
neg.f16x2 r8095, r8092;
}
{
fma.rn.f16x2 r8097, r6685, r8088, r8095;
}
{
mul.f16x2 r8101, r6685, r8090;
}
{
fma.rn.f16x2 r8104, r6829, r8088, r8101;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8108, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8110, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8112, {low, high};
}
{
mul.f16x2 r8113, r8110, r8112;
}
{
mul.f16x2 r8116, r8084, r8108;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8119, {high, low};
}
{
fma.rn.f16x2 r8121, r8113, r8119, r8116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8125, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8127, {high, high};
}
{
mul.f16x2 r8129, r7149, r8127;
}
{
neg.f16x2 r8132, r8129;
}
{
fma.rn.f16x2 r8134, r7005, r8125, r8132;
}
{
mul.f16x2 r8138, r7005, r8127;
}
{
fma.rn.f16x2 r8141, r7149, r8125, r8138;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8145, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8147, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8149, {low, high};
}
{
mul.f16x2 r8150, r8147, r8149;
}
{
mul.f16x2 r8153, r8121, r8145;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8156, {high, low};
}
{
fma.rn.f16x2 r8158, r8150, r8156, r8153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8162, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8164, {high, high};
}
{
mul.f16x2 r8166, r7469, r8164;
}
{
neg.f16x2 r8169, r8166;
}
{
fma.rn.f16x2 r8171, r7325, r8162, r8169;
}
{
mul.f16x2 r8175, r7325, r8164;
}
{
fma.rn.f16x2 r8178, r7469, r8162, r8175;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8182, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8184, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8186, {low, high};
}
{
mul.f16x2 r8187, r8184, r8186;
}
{
mul.f16x2 r8190, r8158, r8182;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8193, {high, low};
}
{
fma.rn.f16x2 r8195, r8187, r8193, r8190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8201, {high, high};
}
{
mul.f16x2 r8203, r7789, r8201;
}
{
neg.f16x2 r8206, r8203;
}
{
fma.rn.f16x2 r8208, r7645, r8199, r8206;
}
{
mul.f16x2 r8212, r7645, r8201;
}
{
fma.rn.f16x2 r8215, r7789, r8199, r8212;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8219, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8221, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8223, {low, high};
}
{
mul.f16x2 r8224, r8221, r8223;
}
{
mul.f16x2 r8227, r8195, r8219;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8230, {high, low};
}
{
fma.rn.f16x2 r8232, r8224, r8230, r8227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8236, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8238, {high, high};
}
{
mul.f16x2 r8240, r6581, r8238;
}
{
neg.f16x2 r8243, r8240;
}
{
fma.rn.f16x2 r8245, r6437, r8236, r8243;
}
{
mul.f16x2 r8249, r6437, r8238;
}
{
fma.rn.f16x2 r8252, r6581, r8236, r8249;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8256, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8258, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8260, {low, high};
}
{
mul.f16x2 r8261, r8258, r8260;
}
{
mul.f16x2 r8264, r8232, r8256;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8267, {high, low};
}
{
fma.rn.f16x2 r8269, r8261, r8267, r8264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8275, {high, high};
}
{
mul.f16x2 r8277, r6901, r8275;
}
{
neg.f16x2 r8280, r8277;
}
{
fma.rn.f16x2 r8282, r6757, r8273, r8280;
}
{
mul.f16x2 r8286, r6757, r8275;
}
{
fma.rn.f16x2 r8289, r6901, r8273, r8286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8293, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8295, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8297, {low, high};
}
{
mul.f16x2 r8298, r8295, r8297;
}
{
mul.f16x2 r8301, r8269, r8293;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8304, {high, low};
}
{
fma.rn.f16x2 r8306, r8298, r8304, r8301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8310, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8312, {high, high};
}
{
mul.f16x2 r8314, r7221, r8312;
}
{
neg.f16x2 r8317, r8314;
}
{
fma.rn.f16x2 r8319, r7077, r8310, r8317;
}
{
mul.f16x2 r8323, r7077, r8312;
}
{
fma.rn.f16x2 r8326, r7221, r8310, r8323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8330, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8332, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8334, {low, high};
}
{
mul.f16x2 r8335, r8332, r8334;
}
{
mul.f16x2 r8338, r8306, r8330;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8341, {high, low};
}
{
fma.rn.f16x2 r8343, r8335, r8341, r8338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8347, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8349, {high, high};
}
{
mul.f16x2 r8351, r7541, r8349;
}
{
neg.f16x2 r8354, r8351;
}
{
fma.rn.f16x2 r8356, r7397, r8347, r8354;
}
{
mul.f16x2 r8360, r7397, r8349;
}
{
fma.rn.f16x2 r8363, r7541, r8347, r8360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8367, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8369, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8371, {low, high};
}
{
mul.f16x2 r8372, r8369, r8371;
}
{
mul.f16x2 r8375, r8343, r8367;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8378, {high, low};
}
{
fma.rn.f16x2 r8380, r8372, r8378, r8375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8384, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8386, {high, high};
}
{
mul.f16x2 r8388, r7861, r8386;
}
{
neg.f16x2 r8391, r8388;
}
{
fma.rn.f16x2 r8393, r7717, r8384, r8391;
}
{
mul.f16x2 r8397, r7717, r8386;
}
{
fma.rn.f16x2 r8400, r7861, r8384, r8397;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8404, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8406, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8408, {low, high};
}
{
mul.f16x2 r8409, r8406, r8408;
}
{
mul.f16x2 r8412, r8380, r8404;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8415, {high, low};
}
{
fma.rn.f16x2 r8417, r8409, r8415, r8412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8421, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8423, {high, high};
}
{
mul.f16x2 r8425, r6617, r8423;
}
{
neg.f16x2 r8428, r8425;
}
{
fma.rn.f16x2 r8430, r6473, r8421, r8428;
}
{
mul.f16x2 r8434, r6473, r8423;
}
{
fma.rn.f16x2 r8437, r6617, r8421, r8434;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8441, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8443, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8445, {low, high};
}
{
mul.f16x2 r8446, r8443, r8445;
}
{
mul.f16x2 r8449, r8417, r8441;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8452, {high, low};
}
{
fma.rn.f16x2 r8454, r8446, r8452, r8449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8458, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8460, {high, high};
}
{
mul.f16x2 r8462, r6937, r8460;
}
{
neg.f16x2 r8465, r8462;
}
{
fma.rn.f16x2 r8467, r6793, r8458, r8465;
}
{
mul.f16x2 r8471, r6793, r8460;
}
{
fma.rn.f16x2 r8474, r6937, r8458, r8471;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8478, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8480, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8482, {low, high};
}
{
mul.f16x2 r8483, r8480, r8482;
}
{
mul.f16x2 r8486, r8454, r8478;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8489, {high, low};
}
{
fma.rn.f16x2 r8491, r8483, r8489, r8486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8495, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8497, {high, high};
}
{
mul.f16x2 r8499, r7257, r8497;
}
{
neg.f16x2 r8502, r8499;
}
{
fma.rn.f16x2 r8504, r7113, r8495, r8502;
}
{
mul.f16x2 r8508, r7113, r8497;
}
{
fma.rn.f16x2 r8511, r7257, r8495, r8508;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8515, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8517, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8519, {low, high};
}
{
mul.f16x2 r8520, r8517, r8519;
}
{
mul.f16x2 r8523, r8491, r8515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8526, {high, low};
}
{
fma.rn.f16x2 r8528, r8520, r8526, r8523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8532, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8534, {high, high};
}
{
mul.f16x2 r8536, r7577, r8534;
}
{
neg.f16x2 r8539, r8536;
}
{
fma.rn.f16x2 r8541, r7433, r8532, r8539;
}
{
mul.f16x2 r8545, r7433, r8534;
}
{
fma.rn.f16x2 r8548, r7577, r8532, r8545;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8552, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8554, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8556, {low, high};
}
{
mul.f16x2 r8557, r8554, r8556;
}
{
mul.f16x2 r8560, r8528, r8552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8563, {high, low};
}
{
fma.rn.f16x2 r8565, r8557, r8563, r8560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8569, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8571, {high, high};
}
{
mul.f16x2 r8573, r7897, r8571;
}
{
neg.f16x2 r8576, r8573;
}
{
fma.rn.f16x2 r8578, r7753, r8569, r8576;
}
{
mul.f16x2 r8582, r7753, r8571;
}
{
fma.rn.f16x2 r8585, r7897, r8569, r8582;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8589, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8591, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8593, {low, high};
}
{
mul.f16x2 r8594, r8591, r8593;
}
{
mul.f16x2 r8597, r8565, r8589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8600, {high, low};
}
{
fma.rn.f16x2 r8602, r8594, r8600, r8597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8606, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8608, {high, high};
}
{
mul.f16x2 r8610, r6545, r8608;
}
{
neg.f16x2 r8613, r8610;
}
{
fma.rn.f16x2 r8615, r6401, r8606, r8613;
}
{
mul.f16x2 r8619, r6401, r8608;
}
{
fma.rn.f16x2 r8622, r6545, r8606, r8619;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8626, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8628, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8630, {low, high};
}
{
mul.f16x2 r8631, r8628, r8630;
}
{
mul.f16x2 r8634, r8602, r8626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8637, {high, low};
}
{
fma.rn.f16x2 r8639, r8631, r8637, r8634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8643, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8645, {high, high};
}
{
mul.f16x2 r8647, r6865, r8645;
}
{
neg.f16x2 r8650, r8647;
}
{
fma.rn.f16x2 r8652, r6721, r8643, r8650;
}
{
mul.f16x2 r8656, r6721, r8645;
}
{
fma.rn.f16x2 r8659, r6865, r8643, r8656;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8663, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8665, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8667, {low, high};
}
{
mul.f16x2 r8668, r8665, r8667;
}
{
mul.f16x2 r8671, r8639, r8663;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8674, {high, low};
}
{
fma.rn.f16x2 r8676, r8668, r8674, r8671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8680, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8682, {high, high};
}
{
mul.f16x2 r8684, r7185, r8682;
}
{
neg.f16x2 r8687, r8684;
}
{
fma.rn.f16x2 r8689, r7041, r8680, r8687;
}
{
mul.f16x2 r8693, r7041, r8682;
}
{
fma.rn.f16x2 r8696, r7185, r8680, r8693;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8702, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8704, {low, high};
}
{
mul.f16x2 r8705, r8702, r8704;
}
{
mul.f16x2 r8708, r8676, r8700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8711, {high, low};
}
{
fma.rn.f16x2 r8713, r8705, r8711, r8708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8717, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8719, {high, high};
}
{
mul.f16x2 r8721, r7505, r8719;
}
{
neg.f16x2 r8724, r8721;
}
{
fma.rn.f16x2 r8726, r7361, r8717, r8724;
}
{
mul.f16x2 r8730, r7361, r8719;
}
{
fma.rn.f16x2 r8733, r7505, r8717, r8730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8739, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8741, {low, high};
}
{
mul.f16x2 r8742, r8739, r8741;
}
{
mul.f16x2 r8745, r8713, r8737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8748, {high, low};
}
{
fma.rn.f16x2 r8750, r8742, r8748, r8745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8750;
mov.b32 r8754, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8750;
mov.b32 r8756, {high, high};
}
{
mul.f16x2 r8758, r7825, r8756;
}
{
neg.f16x2 r8761, r8758;
}
{
fma.rn.f16x2 r8763, r7681, r8754, r8761;
}
{
mul.f16x2 r8767, r7681, r8756;
}
{
fma.rn.f16x2 r8770, r7825, r8754, r8767;
}
shl.b32 r10404, r10403, 3;
add.s32 r10405, r10398, r10404;
barrier.sync 0;
mad.lo.s32 r10406, r10401, 5000, r10405;
st.shared.u32 [r10406], r6317;
st.shared.u32 [r10406+4], r6329;
st.shared.u32 [r10406+200], r7912;
st.shared.u32 [r10406+204], r7919;
st.shared.u32 [r10406+400], r7949;
st.shared.u32 [r10406+404], r7956;
st.shared.u32 [r10406+600], r7986;
st.shared.u32 [r10406+604], r7993;
st.shared.u32 [r10406+800], r8023;
st.shared.u32 [r10406+804], r8030;
st.shared.u32 [r10406+1000], r8060;
st.shared.u32 [r10406+1004], r8067;
st.shared.u32 [r10406+1200], r8097;
st.shared.u32 [r10406+1204], r8104;
st.shared.u32 [r10406+1400], r8134;
st.shared.u32 [r10406+1404], r8141;
st.shared.u32 [r10406+1600], r8171;
st.shared.u32 [r10406+1604], r8178;
st.shared.u32 [r10406+1800], r8208;
st.shared.u32 [r10406+1804], r8215;
st.shared.u32 [r10406+2000], r8245;
st.shared.u32 [r10406+2004], r8252;
st.shared.u32 [r10406+2200], r8282;
st.shared.u32 [r10406+2204], r8289;
st.shared.u32 [r10406+2400], r8319;
st.shared.u32 [r10406+2404], r8326;
st.shared.u32 [r10406+2600], r8356;
st.shared.u32 [r10406+2604], r8363;
st.shared.u32 [r10406+2800], r8393;
st.shared.u32 [r10406+2804], r8400;
st.shared.u32 [r10406+3000], r8430;
st.shared.u32 [r10406+3004], r8437;
st.shared.u32 [r10406+3200], r8467;
st.shared.u32 [r10406+3204], r8474;
st.shared.u32 [r10406+3400], r8504;
st.shared.u32 [r10406+3404], r8511;
st.shared.u32 [r10406+3600], r8541;
st.shared.u32 [r10406+3604], r8548;
st.shared.u32 [r10406+3800], r8578;
st.shared.u32 [r10406+3804], r8585;
st.shared.u32 [r10406+4000], r8615;
st.shared.u32 [r10406+4004], r8622;
st.shared.u32 [r10406+4200], r8652;
st.shared.u32 [r10406+4204], r8659;
st.shared.u32 [r10406+4400], r8689;
st.shared.u32 [r10406+4404], r8696;
st.shared.u32 [r10406+4600], r8726;
st.shared.u32 [r10406+4604], r8733;
st.shared.u32 [r10406+4800], r8763;
st.shared.u32 [r10406+4804], r8770;
barrier.sync 0;
ld.shared.u32 r8803, [r10400];
ld.shared.u32 r8815, [r10400+4];
ld.shared.u32 r9123, [r10400+1000];
ld.shared.u32 r9135, [r10400+1004];
ld.shared.u32 r9443, [r10400+2000];
ld.shared.u32 r9455, [r10400+2004];
ld.shared.u32 r9763, [r10400+3000];
ld.shared.u32 r9775, [r10400+3004];
ld.shared.u32 r10083, [r10400+4000];
ld.shared.u32 r10095, [r10400+4004];
ld.shared.u32 r8800, [r10400+5000];
ld.shared.u32 r8812, [r10400+5004];
ld.shared.u32 r9120, [r10400+6000];
ld.shared.u32 r9132, [r10400+6004];
ld.shared.u32 r9440, [r10400+7000];
ld.shared.u32 r9452, [r10400+7004];
ld.shared.u32 r9760, [r10400+8000];
ld.shared.u32 r9772, [r10400+8004];
ld.shared.u32 r10080, [r10400+9000];
ld.shared.u32 r10092, [r10400+9004];
ld.shared.u32 r8806, [r10400+10000];
ld.shared.u32 r8818, [r10400+10004];
ld.shared.u32 r9126, [r10400+11000];
ld.shared.u32 r9138, [r10400+11004];
ld.shared.u32 r9446, [r10400+12000];
ld.shared.u32 r9458, [r10400+12004];
ld.shared.u32 r9766, [r10400+13000];
ld.shared.u32 r9778, [r10400+13004];
ld.shared.u32 r10086, [r10400+14000];
ld.shared.u32 r10098, [r10400+14004];
ld.shared.u32 r8807, [r10400+15000];
ld.shared.u32 r8819, [r10400+15004];
ld.shared.u32 r9127, [r10400+16000];
ld.shared.u32 r9139, [r10400+16004];
ld.shared.u32 r9447, [r10400+17000];
ld.shared.u32 r9459, [r10400+17004];
ld.shared.u32 r9767, [r10400+18000];
ld.shared.u32 r9779, [r10400+18004];
ld.shared.u32 r10087, [r10400+19000];
ld.shared.u32 r10099, [r10400+19004];
ld.shared.u32 r8801, [r10400+20000];
ld.shared.u32 r8813, [r10400+20004];
ld.shared.u32 r9121, [r10400+21000];
ld.shared.u32 r9133, [r10400+21004];
ld.shared.u32 r9441, [r10400+22000];
ld.shared.u32 r9453, [r10400+22004];
ld.shared.u32 r9761, [r10400+23000];
ld.shared.u32 r9773, [r10400+23004];
ld.shared.u32 r10081, [r10400+24000];
ld.shared.u32 r10093, [r10400+24004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r8793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r8794, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8795, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8796, {low, high};
}
{
neg.f16x2 r8797, r8796;
}
{
add.f16x2 r8799, r8800, r8801;
}
{
add.f16x2 r8802, r8803, r8799;
}
{
add.f16x2 r8805, r8806, r8807;
}
{
add.f16x2 %0, r8802, r8805;
}
{
add.f16x2 r8811, r8812, r8813;
}
{
add.f16x2 r8814, r8815, r8811;
}
{
add.f16x2 r8817, r8818, r8819;
}
{
add.f16x2 %1, r8814, r8817;
}
{
add.f16x2 r8823, r8800, r8801;
}
{
mul.f16x2 r8826, r8823, r8791;
}
{
add.f16x2 r8829, r8803, r8826;
}
{
add.f16x2 r8832, r8806, r8807;
}
{
mul.f16x2 r8835, r8832, r8793;
}
{
add.f16x2 r8838, r8829, r8835;
}
{
sub.f16x2 r8841, r8812, r8813;
}
{
mul.f16x2 r8844, r8841, r8792;
}
{
sub.f16x2 r8847, r8818, r8819;
}
{
mul.f16x2 r8850, r8847, r8794;
}
{
add.f16x2 r8853, r8844, r8850;
}
{
sub.f16x2 %10, r8838, r8853;
}
{
add.f16x2 r8859, r8800, r8801;
}
{
mul.f16x2 r8862, r8859, r8791;
}
{
add.f16x2 r8865, r8803, r8862;
}
{
add.f16x2 r8868, r8806, r8807;
}
{
mul.f16x2 r8871, r8868, r8793;
}
{
add.f16x2 r8874, r8865, r8871;
}
{
sub.f16x2 r8877, r8812, r8813;
}
{
mul.f16x2 r8880, r8877, r8792;
}
{
sub.f16x2 r8883, r8818, r8819;
}
{
mul.f16x2 r8886, r8883, r8794;
}
{
add.f16x2 r8889, r8880, r8886;
}
{
add.f16x2 %40, r8874, r8889;
}
{
add.f16x2 r8895, r8800, r8801;
}
{
mul.f16x2 r8898, r8895, r8793;
}
{
add.f16x2 r8901, r8803, r8898;
}
{
add.f16x2 r8904, r8806, r8807;
}
{
mul.f16x2 r8907, r8904, r8795;
}
{
add.f16x2 r8910, r8901, r8907;
}
{
sub.f16x2 r8913, r8812, r8813;
}
{
mul.f16x2 r8916, r8913, r8794;
}
{
sub.f16x2 r8919, r8818, r8819;
}
{
mul.f16x2 r8922, r8919, r8797;
}
{
add.f16x2 r8925, r8916, r8922;
}
{
sub.f16x2 %20, r8910, r8925;
}
{
add.f16x2 r8931, r8800, r8801;
}
{
mul.f16x2 r8934, r8931, r8793;
}
{
add.f16x2 r8937, r8803, r8934;
}
{
add.f16x2 r8940, r8806, r8807;
}
{
mul.f16x2 r8943, r8940, r8795;
}
{
add.f16x2 r8946, r8937, r8943;
}
{
sub.f16x2 r8949, r8812, r8813;
}
{
mul.f16x2 r8952, r8949, r8794;
}
{
sub.f16x2 r8955, r8818, r8819;
}
{
mul.f16x2 r8958, r8955, r8797;
}
{
add.f16x2 r8961, r8952, r8958;
}
{
add.f16x2 %30, r8946, r8961;
}
{
add.f16x2 r8967, r8812, r8813;
}
{
mul.f16x2 r8970, r8967, r8791;
}
{
add.f16x2 r8973, r8815, r8970;
}
{
add.f16x2 r8976, r8818, r8819;
}
{
mul.f16x2 r8979, r8976, r8793;
}
{
add.f16x2 r8982, r8973, r8979;
}
{
sub.f16x2 r8985, r8800, r8801;
}
{
mul.f16x2 r8988, r8985, r8792;
}
{
sub.f16x2 r8991, r8806, r8807;
}
{
mul.f16x2 r8994, r8991, r8794;
}
{
add.f16x2 r8997, r8988, r8994;
}
{
add.f16x2 %11, r8982, r8997;
}
{
add.f16x2 r9003, r8812, r8813;
}
{
mul.f16x2 r9006, r9003, r8791;
}
{
add.f16x2 r9009, r8815, r9006;
}
{
add.f16x2 r9012, r8818, r8819;
}
{
mul.f16x2 r9015, r9012, r8793;
}
{
add.f16x2 r9018, r9009, r9015;
}
{
sub.f16x2 r9021, r8800, r8801;
}
{
mul.f16x2 r9024, r9021, r8792;
}
{
sub.f16x2 r9027, r8806, r8807;
}
{
mul.f16x2 r9030, r9027, r8794;
}
{
add.f16x2 r9033, r9024, r9030;
}
{
sub.f16x2 %41, r9018, r9033;
}
{
add.f16x2 r9039, r8812, r8813;
}
{
mul.f16x2 r9042, r9039, r8793;
}
{
add.f16x2 r9045, r8815, r9042;
}
{
add.f16x2 r9048, r8818, r8819;
}
{
mul.f16x2 r9051, r9048, r8795;
}
{
add.f16x2 r9054, r9045, r9051;
}
{
sub.f16x2 r9057, r8800, r8801;
}
{
mul.f16x2 r9060, r9057, r8794;
}
{
sub.f16x2 r9063, r8806, r8807;
}
{
mul.f16x2 r9066, r9063, r8797;
}
{
add.f16x2 r9069, r9060, r9066;
}
{
add.f16x2 %21, r9054, r9069;
}
{
add.f16x2 r9075, r8812, r8813;
}
{
mul.f16x2 r9078, r9075, r8793;
}
{
add.f16x2 r9081, r8815, r9078;
}
{
add.f16x2 r9084, r8818, r8819;
}
{
mul.f16x2 r9087, r9084, r8795;
}
{
add.f16x2 r9090, r9081, r9087;
}
{
sub.f16x2 r9093, r8800, r8801;
}
{
mul.f16x2 r9096, r9093, r8794;
}
{
sub.f16x2 r9099, r8806, r8807;
}
{
mul.f16x2 r9102, r9099, r8797;
}
{
add.f16x2 r9105, r9096, r9102;
}
{
sub.f16x2 %31, r9090, r9105;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9116, {low, high};
}
{
neg.f16x2 r9117, r9116;
}
{
add.f16x2 r9119, r9120, r9121;
}
{
add.f16x2 r9122, r9123, r9119;
}
{
add.f16x2 r9125, r9126, r9127;
}
{
add.f16x2 %2, r9122, r9125;
}
{
add.f16x2 r9131, r9132, r9133;
}
{
add.f16x2 r9134, r9135, r9131;
}
{
add.f16x2 r9137, r9138, r9139;
}
{
add.f16x2 %3, r9134, r9137;
}
{
add.f16x2 r9143, r9120, r9121;
}
{
mul.f16x2 r9146, r9143, r9111;
}
{
add.f16x2 r9149, r9123, r9146;
}
{
add.f16x2 r9152, r9126, r9127;
}
{
mul.f16x2 r9155, r9152, r9113;
}
{
add.f16x2 r9158, r9149, r9155;
}
{
sub.f16x2 r9161, r9132, r9133;
}
{
mul.f16x2 r9164, r9161, r9112;
}
{
sub.f16x2 r9167, r9138, r9139;
}
{
mul.f16x2 r9170, r9167, r9114;
}
{
add.f16x2 r9173, r9164, r9170;
}
{
sub.f16x2 %12, r9158, r9173;
}
{
add.f16x2 r9179, r9120, r9121;
}
{
mul.f16x2 r9182, r9179, r9111;
}
{
add.f16x2 r9185, r9123, r9182;
}
{
add.f16x2 r9188, r9126, r9127;
}
{
mul.f16x2 r9191, r9188, r9113;
}
{
add.f16x2 r9194, r9185, r9191;
}
{
sub.f16x2 r9197, r9132, r9133;
}
{
mul.f16x2 r9200, r9197, r9112;
}
{
sub.f16x2 r9203, r9138, r9139;
}
{
mul.f16x2 r9206, r9203, r9114;
}
{
add.f16x2 r9209, r9200, r9206;
}
{
add.f16x2 %42, r9194, r9209;
}
{
add.f16x2 r9215, r9120, r9121;
}
{
mul.f16x2 r9218, r9215, r9113;
}
{
add.f16x2 r9221, r9123, r9218;
}
{
add.f16x2 r9224, r9126, r9127;
}
{
mul.f16x2 r9227, r9224, r9115;
}
{
add.f16x2 r9230, r9221, r9227;
}
{
sub.f16x2 r9233, r9132, r9133;
}
{
mul.f16x2 r9236, r9233, r9114;
}
{
sub.f16x2 r9239, r9138, r9139;
}
{
mul.f16x2 r9242, r9239, r9117;
}
{
add.f16x2 r9245, r9236, r9242;
}
{
sub.f16x2 %22, r9230, r9245;
}
{
add.f16x2 r9251, r9120, r9121;
}
{
mul.f16x2 r9254, r9251, r9113;
}
{
add.f16x2 r9257, r9123, r9254;
}
{
add.f16x2 r9260, r9126, r9127;
}
{
mul.f16x2 r9263, r9260, r9115;
}
{
add.f16x2 r9266, r9257, r9263;
}
{
sub.f16x2 r9269, r9132, r9133;
}
{
mul.f16x2 r9272, r9269, r9114;
}
{
sub.f16x2 r9275, r9138, r9139;
}
{
mul.f16x2 r9278, r9275, r9117;
}
{
add.f16x2 r9281, r9272, r9278;
}
{
add.f16x2 %32, r9266, r9281;
}
{
add.f16x2 r9287, r9132, r9133;
}
{
mul.f16x2 r9290, r9287, r9111;
}
{
add.f16x2 r9293, r9135, r9290;
}
{
add.f16x2 r9296, r9138, r9139;
}
{
mul.f16x2 r9299, r9296, r9113;
}
{
add.f16x2 r9302, r9293, r9299;
}
{
sub.f16x2 r9305, r9120, r9121;
}
{
mul.f16x2 r9308, r9305, r9112;
}
{
sub.f16x2 r9311, r9126, r9127;
}
{
mul.f16x2 r9314, r9311, r9114;
}
{
add.f16x2 r9317, r9308, r9314;
}
{
add.f16x2 %13, r9302, r9317;
}
{
add.f16x2 r9323, r9132, r9133;
}
{
mul.f16x2 r9326, r9323, r9111;
}
{
add.f16x2 r9329, r9135, r9326;
}
{
add.f16x2 r9332, r9138, r9139;
}
{
mul.f16x2 r9335, r9332, r9113;
}
{
add.f16x2 r9338, r9329, r9335;
}
{
sub.f16x2 r9341, r9120, r9121;
}
{
mul.f16x2 r9344, r9341, r9112;
}
{
sub.f16x2 r9347, r9126, r9127;
}
{
mul.f16x2 r9350, r9347, r9114;
}
{
add.f16x2 r9353, r9344, r9350;
}
{
sub.f16x2 %43, r9338, r9353;
}
{
add.f16x2 r9359, r9132, r9133;
}
{
mul.f16x2 r9362, r9359, r9113;
}
{
add.f16x2 r9365, r9135, r9362;
}
{
add.f16x2 r9368, r9138, r9139;
}
{
mul.f16x2 r9371, r9368, r9115;
}
{
add.f16x2 r9374, r9365, r9371;
}
{
sub.f16x2 r9377, r9120, r9121;
}
{
mul.f16x2 r9380, r9377, r9114;
}
{
sub.f16x2 r9383, r9126, r9127;
}
{
mul.f16x2 r9386, r9383, r9117;
}
{
add.f16x2 r9389, r9380, r9386;
}
{
add.f16x2 %23, r9374, r9389;
}
{
add.f16x2 r9395, r9132, r9133;
}
{
mul.f16x2 r9398, r9395, r9113;
}
{
add.f16x2 r9401, r9135, r9398;
}
{
add.f16x2 r9404, r9138, r9139;
}
{
mul.f16x2 r9407, r9404, r9115;
}
{
add.f16x2 r9410, r9401, r9407;
}
{
sub.f16x2 r9413, r9120, r9121;
}
{
mul.f16x2 r9416, r9413, r9114;
}
{
sub.f16x2 r9419, r9126, r9127;
}
{
mul.f16x2 r9422, r9419, r9117;
}
{
add.f16x2 r9425, r9416, r9422;
}
{
sub.f16x2 %33, r9410, r9425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9434, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9435, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9436, {low, high};
}
{
neg.f16x2 r9437, r9436;
}
{
add.f16x2 r9439, r9440, r9441;
}
{
add.f16x2 r9442, r9443, r9439;
}
{
add.f16x2 r9445, r9446, r9447;
}
{
add.f16x2 %4, r9442, r9445;
}
{
add.f16x2 r9451, r9452, r9453;
}
{
add.f16x2 r9454, r9455, r9451;
}
{
add.f16x2 r9457, r9458, r9459;
}
{
add.f16x2 %5, r9454, r9457;
}
{
add.f16x2 r9463, r9440, r9441;
}
{
mul.f16x2 r9466, r9463, r9431;
}
{
add.f16x2 r9469, r9443, r9466;
}
{
add.f16x2 r9472, r9446, r9447;
}
{
mul.f16x2 r9475, r9472, r9433;
}
{
add.f16x2 r9478, r9469, r9475;
}
{
sub.f16x2 r9481, r9452, r9453;
}
{
mul.f16x2 r9484, r9481, r9432;
}
{
sub.f16x2 r9487, r9458, r9459;
}
{
mul.f16x2 r9490, r9487, r9434;
}
{
add.f16x2 r9493, r9484, r9490;
}
{
sub.f16x2 %14, r9478, r9493;
}
{
add.f16x2 r9499, r9440, r9441;
}
{
mul.f16x2 r9502, r9499, r9431;
}
{
add.f16x2 r9505, r9443, r9502;
}
{
add.f16x2 r9508, r9446, r9447;
}
{
mul.f16x2 r9511, r9508, r9433;
}
{
add.f16x2 r9514, r9505, r9511;
}
{
sub.f16x2 r9517, r9452, r9453;
}
{
mul.f16x2 r9520, r9517, r9432;
}
{
sub.f16x2 r9523, r9458, r9459;
}
{
mul.f16x2 r9526, r9523, r9434;
}
{
add.f16x2 r9529, r9520, r9526;
}
{
add.f16x2 %44, r9514, r9529;
}
{
add.f16x2 r9535, r9440, r9441;
}
{
mul.f16x2 r9538, r9535, r9433;
}
{
add.f16x2 r9541, r9443, r9538;
}
{
add.f16x2 r9544, r9446, r9447;
}
{
mul.f16x2 r9547, r9544, r9435;
}
{
add.f16x2 r9550, r9541, r9547;
}
{
sub.f16x2 r9553, r9452, r9453;
}
{
mul.f16x2 r9556, r9553, r9434;
}
{
sub.f16x2 r9559, r9458, r9459;
}
{
mul.f16x2 r9562, r9559, r9437;
}
{
add.f16x2 r9565, r9556, r9562;
}
{
sub.f16x2 %24, r9550, r9565;
}
{
add.f16x2 r9571, r9440, r9441;
}
{
mul.f16x2 r9574, r9571, r9433;
}
{
add.f16x2 r9577, r9443, r9574;
}
{
add.f16x2 r9580, r9446, r9447;
}
{
mul.f16x2 r9583, r9580, r9435;
}
{
add.f16x2 r9586, r9577, r9583;
}
{
sub.f16x2 r9589, r9452, r9453;
}
{
mul.f16x2 r9592, r9589, r9434;
}
{
sub.f16x2 r9595, r9458, r9459;
}
{
mul.f16x2 r9598, r9595, r9437;
}
{
add.f16x2 r9601, r9592, r9598;
}
{
add.f16x2 %34, r9586, r9601;
}
{
add.f16x2 r9607, r9452, r9453;
}
{
mul.f16x2 r9610, r9607, r9431;
}
{
add.f16x2 r9613, r9455, r9610;
}
{
add.f16x2 r9616, r9458, r9459;
}
{
mul.f16x2 r9619, r9616, r9433;
}
{
add.f16x2 r9622, r9613, r9619;
}
{
sub.f16x2 r9625, r9440, r9441;
}
{
mul.f16x2 r9628, r9625, r9432;
}
{
sub.f16x2 r9631, r9446, r9447;
}
{
mul.f16x2 r9634, r9631, r9434;
}
{
add.f16x2 r9637, r9628, r9634;
}
{
add.f16x2 %15, r9622, r9637;
}
{
add.f16x2 r9643, r9452, r9453;
}
{
mul.f16x2 r9646, r9643, r9431;
}
{
add.f16x2 r9649, r9455, r9646;
}
{
add.f16x2 r9652, r9458, r9459;
}
{
mul.f16x2 r9655, r9652, r9433;
}
{
add.f16x2 r9658, r9649, r9655;
}
{
sub.f16x2 r9661, r9440, r9441;
}
{
mul.f16x2 r9664, r9661, r9432;
}
{
sub.f16x2 r9667, r9446, r9447;
}
{
mul.f16x2 r9670, r9667, r9434;
}
{
add.f16x2 r9673, r9664, r9670;
}
{
sub.f16x2 %45, r9658, r9673;
}
{
add.f16x2 r9679, r9452, r9453;
}
{
mul.f16x2 r9682, r9679, r9433;
}
{
add.f16x2 r9685, r9455, r9682;
}
{
add.f16x2 r9688, r9458, r9459;
}
{
mul.f16x2 r9691, r9688, r9435;
}
{
add.f16x2 r9694, r9685, r9691;
}
{
sub.f16x2 r9697, r9440, r9441;
}
{
mul.f16x2 r9700, r9697, r9434;
}
{
sub.f16x2 r9703, r9446, r9447;
}
{
mul.f16x2 r9706, r9703, r9437;
}
{
add.f16x2 r9709, r9700, r9706;
}
{
add.f16x2 %25, r9694, r9709;
}
{
add.f16x2 r9715, r9452, r9453;
}
{
mul.f16x2 r9718, r9715, r9433;
}
{
add.f16x2 r9721, r9455, r9718;
}
{
add.f16x2 r9724, r9458, r9459;
}
{
mul.f16x2 r9727, r9724, r9435;
}
{
add.f16x2 r9730, r9721, r9727;
}
{
sub.f16x2 r9733, r9440, r9441;
}
{
mul.f16x2 r9736, r9733, r9434;
}
{
sub.f16x2 r9739, r9446, r9447;
}
{
mul.f16x2 r9742, r9739, r9437;
}
{
add.f16x2 r9745, r9736, r9742;
}
{
sub.f16x2 %35, r9730, r9745;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9751, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9752, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9753, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9754, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9755, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9756, {low, high};
}
{
neg.f16x2 r9757, r9756;
}
{
add.f16x2 r9759, r9760, r9761;
}
{
add.f16x2 r9762, r9763, r9759;
}
{
add.f16x2 r9765, r9766, r9767;
}
{
add.f16x2 %6, r9762, r9765;
}
{
add.f16x2 r9771, r9772, r9773;
}
{
add.f16x2 r9774, r9775, r9771;
}
{
add.f16x2 r9777, r9778, r9779;
}
{
add.f16x2 %7, r9774, r9777;
}
{
add.f16x2 r9783, r9760, r9761;
}
{
mul.f16x2 r9786, r9783, r9751;
}
{
add.f16x2 r9789, r9763, r9786;
}
{
add.f16x2 r9792, r9766, r9767;
}
{
mul.f16x2 r9795, r9792, r9753;
}
{
add.f16x2 r9798, r9789, r9795;
}
{
sub.f16x2 r9801, r9772, r9773;
}
{
mul.f16x2 r9804, r9801, r9752;
}
{
sub.f16x2 r9807, r9778, r9779;
}
{
mul.f16x2 r9810, r9807, r9754;
}
{
add.f16x2 r9813, r9804, r9810;
}
{
sub.f16x2 %16, r9798, r9813;
}
{
add.f16x2 r9819, r9760, r9761;
}
{
mul.f16x2 r9822, r9819, r9751;
}
{
add.f16x2 r9825, r9763, r9822;
}
{
add.f16x2 r9828, r9766, r9767;
}
{
mul.f16x2 r9831, r9828, r9753;
}
{
add.f16x2 r9834, r9825, r9831;
}
{
sub.f16x2 r9837, r9772, r9773;
}
{
mul.f16x2 r9840, r9837, r9752;
}
{
sub.f16x2 r9843, r9778, r9779;
}
{
mul.f16x2 r9846, r9843, r9754;
}
{
add.f16x2 r9849, r9840, r9846;
}
{
add.f16x2 %46, r9834, r9849;
}
{
add.f16x2 r9855, r9760, r9761;
}
{
mul.f16x2 r9858, r9855, r9753;
}
{
add.f16x2 r9861, r9763, r9858;
}
{
add.f16x2 r9864, r9766, r9767;
}
{
mul.f16x2 r9867, r9864, r9755;
}
{
add.f16x2 r9870, r9861, r9867;
}
{
sub.f16x2 r9873, r9772, r9773;
}
{
mul.f16x2 r9876, r9873, r9754;
}
{
sub.f16x2 r9879, r9778, r9779;
}
{
mul.f16x2 r9882, r9879, r9757;
}
{
add.f16x2 r9885, r9876, r9882;
}
{
sub.f16x2 %26, r9870, r9885;
}
{
add.f16x2 r9891, r9760, r9761;
}
{
mul.f16x2 r9894, r9891, r9753;
}
{
add.f16x2 r9897, r9763, r9894;
}
{
add.f16x2 r9900, r9766, r9767;
}
{
mul.f16x2 r9903, r9900, r9755;
}
{
add.f16x2 r9906, r9897, r9903;
}
{
sub.f16x2 r9909, r9772, r9773;
}
{
mul.f16x2 r9912, r9909, r9754;
}
{
sub.f16x2 r9915, r9778, r9779;
}
{
mul.f16x2 r9918, r9915, r9757;
}
{
add.f16x2 r9921, r9912, r9918;
}
{
add.f16x2 %36, r9906, r9921;
}
{
add.f16x2 r9927, r9772, r9773;
}
{
mul.f16x2 r9930, r9927, r9751;
}
{
add.f16x2 r9933, r9775, r9930;
}
{
add.f16x2 r9936, r9778, r9779;
}
{
mul.f16x2 r9939, r9936, r9753;
}
{
add.f16x2 r9942, r9933, r9939;
}
{
sub.f16x2 r9945, r9760, r9761;
}
{
mul.f16x2 r9948, r9945, r9752;
}
{
sub.f16x2 r9951, r9766, r9767;
}
{
mul.f16x2 r9954, r9951, r9754;
}
{
add.f16x2 r9957, r9948, r9954;
}
{
add.f16x2 %17, r9942, r9957;
}
{
add.f16x2 r9963, r9772, r9773;
}
{
mul.f16x2 r9966, r9963, r9751;
}
{
add.f16x2 r9969, r9775, r9966;
}
{
add.f16x2 r9972, r9778, r9779;
}
{
mul.f16x2 r9975, r9972, r9753;
}
{
add.f16x2 r9978, r9969, r9975;
}
{
sub.f16x2 r9981, r9760, r9761;
}
{
mul.f16x2 r9984, r9981, r9752;
}
{
sub.f16x2 r9987, r9766, r9767;
}
{
mul.f16x2 r9990, r9987, r9754;
}
{
add.f16x2 r9993, r9984, r9990;
}
{
sub.f16x2 %47, r9978, r9993;
}
{
add.f16x2 r9999, r9772, r9773;
}
{
mul.f16x2 r10002, r9999, r9753;
}
{
add.f16x2 r10005, r9775, r10002;
}
{
add.f16x2 r10008, r9778, r9779;
}
{
mul.f16x2 r10011, r10008, r9755;
}
{
add.f16x2 r10014, r10005, r10011;
}
{
sub.f16x2 r10017, r9760, r9761;
}
{
mul.f16x2 r10020, r10017, r9754;
}
{
sub.f16x2 r10023, r9766, r9767;
}
{
mul.f16x2 r10026, r10023, r9757;
}
{
add.f16x2 r10029, r10020, r10026;
}
{
add.f16x2 %27, r10014, r10029;
}
{
add.f16x2 r10035, r9772, r9773;
}
{
mul.f16x2 r10038, r10035, r9753;
}
{
add.f16x2 r10041, r9775, r10038;
}
{
add.f16x2 r10044, r9778, r9779;
}
{
mul.f16x2 r10047, r10044, r9755;
}
{
add.f16x2 r10050, r10041, r10047;
}
{
sub.f16x2 r10053, r9760, r9761;
}
{
mul.f16x2 r10056, r10053, r9754;
}
{
sub.f16x2 r10059, r9766, r9767;
}
{
mul.f16x2 r10062, r10059, r9757;
}
{
add.f16x2 r10065, r10056, r10062;
}
{
sub.f16x2 %37, r10050, r10065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10071, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10072, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r10073, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r10074, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10075, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10076, {low, high};
}
{
neg.f16x2 r10077, r10076;
}
{
add.f16x2 r10079, r10080, r10081;
}
{
add.f16x2 r10082, r10083, r10079;
}
{
add.f16x2 r10085, r10086, r10087;
}
{
add.f16x2 %8, r10082, r10085;
}
{
add.f16x2 r10091, r10092, r10093;
}
{
add.f16x2 r10094, r10095, r10091;
}
{
add.f16x2 r10097, r10098, r10099;
}
{
add.f16x2 %9, r10094, r10097;
}
{
add.f16x2 r10103, r10080, r10081;
}
{
mul.f16x2 r10106, r10103, r10071;
}
{
add.f16x2 r10109, r10083, r10106;
}
{
add.f16x2 r10112, r10086, r10087;
}
{
mul.f16x2 r10115, r10112, r10073;
}
{
add.f16x2 r10118, r10109, r10115;
}
{
sub.f16x2 r10121, r10092, r10093;
}
{
mul.f16x2 r10124, r10121, r10072;
}
{
sub.f16x2 r10127, r10098, r10099;
}
{
mul.f16x2 r10130, r10127, r10074;
}
{
add.f16x2 r10133, r10124, r10130;
}
{
sub.f16x2 %18, r10118, r10133;
}
{
add.f16x2 r10139, r10080, r10081;
}
{
mul.f16x2 r10142, r10139, r10071;
}
{
add.f16x2 r10145, r10083, r10142;
}
{
add.f16x2 r10148, r10086, r10087;
}
{
mul.f16x2 r10151, r10148, r10073;
}
{
add.f16x2 r10154, r10145, r10151;
}
{
sub.f16x2 r10157, r10092, r10093;
}
{
mul.f16x2 r10160, r10157, r10072;
}
{
sub.f16x2 r10163, r10098, r10099;
}
{
mul.f16x2 r10166, r10163, r10074;
}
{
add.f16x2 r10169, r10160, r10166;
}
{
add.f16x2 %48, r10154, r10169;
}
{
add.f16x2 r10175, r10080, r10081;
}
{
mul.f16x2 r10178, r10175, r10073;
}
{
add.f16x2 r10181, r10083, r10178;
}
{
add.f16x2 r10184, r10086, r10087;
}
{
mul.f16x2 r10187, r10184, r10075;
}
{
add.f16x2 r10190, r10181, r10187;
}
{
sub.f16x2 r10193, r10092, r10093;
}
{
mul.f16x2 r10196, r10193, r10074;
}
{
sub.f16x2 r10199, r10098, r10099;
}
{
mul.f16x2 r10202, r10199, r10077;
}
{
add.f16x2 r10205, r10196, r10202;
}
{
sub.f16x2 %28, r10190, r10205;
}
{
add.f16x2 r10211, r10080, r10081;
}
{
mul.f16x2 r10214, r10211, r10073;
}
{
add.f16x2 r10217, r10083, r10214;
}
{
add.f16x2 r10220, r10086, r10087;
}
{
mul.f16x2 r10223, r10220, r10075;
}
{
add.f16x2 r10226, r10217, r10223;
}
{
sub.f16x2 r10229, r10092, r10093;
}
{
mul.f16x2 r10232, r10229, r10074;
}
{
sub.f16x2 r10235, r10098, r10099;
}
{
mul.f16x2 r10238, r10235, r10077;
}
{
add.f16x2 r10241, r10232, r10238;
}
{
add.f16x2 %38, r10226, r10241;
}
{
add.f16x2 r10247, r10092, r10093;
}
{
mul.f16x2 r10250, r10247, r10071;
}
{
add.f16x2 r10253, r10095, r10250;
}
{
add.f16x2 r10256, r10098, r10099;
}
{
mul.f16x2 r10259, r10256, r10073;
}
{
add.f16x2 r10262, r10253, r10259;
}
{
sub.f16x2 r10265, r10080, r10081;
}
{
mul.f16x2 r10268, r10265, r10072;
}
{
sub.f16x2 r10271, r10086, r10087;
}
{
mul.f16x2 r10274, r10271, r10074;
}
{
add.f16x2 r10277, r10268, r10274;
}
{
add.f16x2 %19, r10262, r10277;
}
{
add.f16x2 r10283, r10092, r10093;
}
{
mul.f16x2 r10286, r10283, r10071;
}
{
add.f16x2 r10289, r10095, r10286;
}
{
add.f16x2 r10292, r10098, r10099;
}
{
mul.f16x2 r10295, r10292, r10073;
}
{
add.f16x2 r10298, r10289, r10295;
}
{
sub.f16x2 r10301, r10080, r10081;
}
{
mul.f16x2 r10304, r10301, r10072;
}
{
sub.f16x2 r10307, r10086, r10087;
}
{
mul.f16x2 r10310, r10307, r10074;
}
{
add.f16x2 r10313, r10304, r10310;
}
{
sub.f16x2 %49, r10298, r10313;
}
{
add.f16x2 r10319, r10092, r10093;
}
{
mul.f16x2 r10322, r10319, r10073;
}
{
add.f16x2 r10325, r10095, r10322;
}
{
add.f16x2 r10328, r10098, r10099;
}
{
mul.f16x2 r10331, r10328, r10075;
}
{
add.f16x2 r10334, r10325, r10331;
}
{
sub.f16x2 r10337, r10080, r10081;
}
{
mul.f16x2 r10340, r10337, r10074;
}
{
sub.f16x2 r10343, r10086, r10087;
}
{
mul.f16x2 r10346, r10343, r10077;
}
{
add.f16x2 r10349, r10340, r10346;
}
{
add.f16x2 %29, r10334, r10349;
}
{
add.f16x2 r10355, r10092, r10093;
}
{
mul.f16x2 r10358, r10355, r10073;
}
{
add.f16x2 r10361, r10095, r10358;
}
{
add.f16x2 r10364, r10098, r10099;
}
{
mul.f16x2 r10367, r10364, r10075;
}
{
add.f16x2 r10370, r10361, r10367;
}
{
sub.f16x2 r10373, r10080, r10081;
}
{
mul.f16x2 r10376, r10373, r10074;
}
{
sub.f16x2 r10379, r10086, r10087;
}
{
mul.f16x2 r10382, r10379, r10077;
}
{
add.f16x2 r10385, r10376, r10382;
}
{
sub.f16x2 %39, r10370, r10385;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[18].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<912, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<603>;
.reg .b32 r<10458>;
.reg .b64 rd<6>;
mov.u32 r10392, %50;
mov.u32 r10457, %tid.y;
mad.lo.s32 r10393, r10457, 12500, r10392;
mov.u32 r10394, %tid.x;
mov.f32 f594, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1, {low, high};
}
mov.f32 f596, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2, {low, high};
}
mov.f32 f590, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3, {low, high};
}
mov.f32 f592, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %95, %81;
}
{
add.f16x2 r12, %54, r9;
}
{
add.f16x2 r15, %59, %94;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %70, %58;
}
{
add.f16x2 r24, %79, r21;
}
{
add.f16x2 r27, %87, %69;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %95, %81;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %54, r36;
}
{
add.f16x2 r42, %59, %94;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %70, %58;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %87, %69;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %95, %81;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %54, r72;
}
{
add.f16x2 r78, %59, %94;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %70, %58;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %87, %69;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %95, %81;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %54, r108;
}
{
add.f16x2 r114, %59, %94;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %70, %58;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %87, %69;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %95, %81;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %54, r144;
}
{
add.f16x2 r150, %59, %94;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %70, %58;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %87, %69;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %70, %58;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %79, r180;
}
{
add.f16x2 r186, %87, %69;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %95, %81;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %59, %94;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %70, %58;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %79, r216;
}
{
add.f16x2 r222, %87, %69;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %95, %81;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %59, %94;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %70, %58;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %79, r252;
}
{
add.f16x2 r258, %87, %69;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %95, %81;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %59, %94;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %70, %58;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %79, r288;
}
{
add.f16x2 r294, %87, %69;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %95, %81;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %59, %94;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r326, {low, high};
}
{
neg.f16x2 r327, r326;
}
{
add.f16x2 r329, %97, %83;
}
{
add.f16x2 r332, %55, r329;
}
{
add.f16x2 r335, %62, %96;
}
{
add.f16x2 r338, r332, r335;
}
{
add.f16x2 r341, %72, %61;
}
{
add.f16x2 r344, %80, r341;
}
{
add.f16x2 r347, %89, %71;
}
{
add.f16x2 r350, r344, r347;
}
{
add.f16x2 r353, %97, %83;
}
{
mul.f16x2 r356, r353, r321;
}
{
add.f16x2 r359, %55, r356;
}
{
add.f16x2 r362, %62, %96;
}
{
mul.f16x2 r365, r362, r323;
}
{
add.f16x2 r368, r359, r365;
}
{
sub.f16x2 r371, %72, %61;
}
{
mul.f16x2 r374, r371, r322;
}
{
sub.f16x2 r377, %89, %71;
}
{
mul.f16x2 r380, r377, r324;
}
{
add.f16x2 r383, r374, r380;
}
{
sub.f16x2 r386, r368, r383;
}
{
add.f16x2 r389, %97, %83;
}
{
mul.f16x2 r392, r389, r321;
}
{
add.f16x2 r395, %55, r392;
}
{
add.f16x2 r398, %62, %96;
}
{
mul.f16x2 r401, r398, r323;
}
{
add.f16x2 r404, r395, r401;
}
{
sub.f16x2 r407, %72, %61;
}
{
mul.f16x2 r410, r407, r322;
}
{
sub.f16x2 r413, %89, %71;
}
{
mul.f16x2 r416, r413, r324;
}
{
add.f16x2 r419, r410, r416;
}
{
add.f16x2 r422, r404, r419;
}
{
add.f16x2 r425, %97, %83;
}
{
mul.f16x2 r428, r425, r323;
}
{
add.f16x2 r431, %55, r428;
}
{
add.f16x2 r434, %62, %96;
}
{
mul.f16x2 r437, r434, r325;
}
{
add.f16x2 r440, r431, r437;
}
{
sub.f16x2 r443, %72, %61;
}
{
mul.f16x2 r446, r443, r324;
}
{
sub.f16x2 r449, %89, %71;
}
{
mul.f16x2 r452, r449, r327;
}
{
add.f16x2 r455, r446, r452;
}
{
sub.f16x2 r458, r440, r455;
}
{
add.f16x2 r461, %97, %83;
}
{
mul.f16x2 r464, r461, r323;
}
{
add.f16x2 r467, %55, r464;
}
{
add.f16x2 r470, %62, %96;
}
{
mul.f16x2 r473, r470, r325;
}
{
add.f16x2 r476, r467, r473;
}
{
sub.f16x2 r479, %72, %61;
}
{
mul.f16x2 r482, r479, r324;
}
{
sub.f16x2 r485, %89, %71;
}
{
mul.f16x2 r488, r485, r327;
}
{
add.f16x2 r491, r482, r488;
}
{
add.f16x2 r494, r476, r491;
}
{
add.f16x2 r497, %72, %61;
}
{
mul.f16x2 r500, r497, r321;
}
{
add.f16x2 r503, %80, r500;
}
{
add.f16x2 r506, %89, %71;
}
{
mul.f16x2 r509, r506, r323;
}
{
add.f16x2 r512, r503, r509;
}
{
sub.f16x2 r515, %97, %83;
}
{
mul.f16x2 r518, r515, r322;
}
{
sub.f16x2 r521, %62, %96;
}
{
mul.f16x2 r524, r521, r324;
}
{
add.f16x2 r527, r518, r524;
}
{
add.f16x2 r530, r512, r527;
}
{
add.f16x2 r533, %72, %61;
}
{
mul.f16x2 r536, r533, r321;
}
{
add.f16x2 r539, %80, r536;
}
{
add.f16x2 r542, %89, %71;
}
{
mul.f16x2 r545, r542, r323;
}
{
add.f16x2 r548, r539, r545;
}
{
sub.f16x2 r551, %97, %83;
}
{
mul.f16x2 r554, r551, r322;
}
{
sub.f16x2 r557, %62, %96;
}
{
mul.f16x2 r560, r557, r324;
}
{
add.f16x2 r563, r554, r560;
}
{
sub.f16x2 r566, r548, r563;
}
{
add.f16x2 r569, %72, %61;
}
{
mul.f16x2 r572, r569, r323;
}
{
add.f16x2 r575, %80, r572;
}
{
add.f16x2 r578, %89, %71;
}
{
mul.f16x2 r581, r578, r325;
}
{
add.f16x2 r584, r575, r581;
}
{
sub.f16x2 r587, %97, %83;
}
{
mul.f16x2 r590, r587, r324;
}
{
sub.f16x2 r593, %62, %96;
}
{
mul.f16x2 r596, r593, r327;
}
{
add.f16x2 r599, r590, r596;
}
{
add.f16x2 r602, r584, r599;
}
{
add.f16x2 r605, %72, %61;
}
{
mul.f16x2 r608, r605, r323;
}
{
add.f16x2 r611, %80, r608;
}
{
add.f16x2 r614, %89, %71;
}
{
mul.f16x2 r617, r614, r325;
}
{
add.f16x2 r620, r611, r617;
}
{
sub.f16x2 r623, %97, %83;
}
{
mul.f16x2 r626, r623, r324;
}
{
sub.f16x2 r629, %62, %96;
}
{
mul.f16x2 r632, r629, r327;
}
{
add.f16x2 r635, r626, r632;
}
{
sub.f16x2 r638, r620, r635;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r644, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r646, {low, high};
}
{
neg.f16x2 r647, r646;
}
{
add.f16x2 r649, %99, %85;
}
{
add.f16x2 r652, %56, r649;
}
{
add.f16x2 r655, %64, %98;
}
{
add.f16x2 r658, r652, r655;
}
{
add.f16x2 r661, %74, %63;
}
{
add.f16x2 r664, %82, r661;
}
{
add.f16x2 r667, %91, %73;
}
{
add.f16x2 r670, r664, r667;
}
{
add.f16x2 r673, %99, %85;
}
{
mul.f16x2 r676, r673, r641;
}
{
add.f16x2 r679, %56, r676;
}
{
add.f16x2 r682, %64, %98;
}
{
mul.f16x2 r685, r682, r643;
}
{
add.f16x2 r688, r679, r685;
}
{
sub.f16x2 r691, %74, %63;
}
{
mul.f16x2 r694, r691, r642;
}
{
sub.f16x2 r697, %91, %73;
}
{
mul.f16x2 r700, r697, r644;
}
{
add.f16x2 r703, r694, r700;
}
{
sub.f16x2 r706, r688, r703;
}
{
add.f16x2 r709, %99, %85;
}
{
mul.f16x2 r712, r709, r641;
}
{
add.f16x2 r715, %56, r712;
}
{
add.f16x2 r718, %64, %98;
}
{
mul.f16x2 r721, r718, r643;
}
{
add.f16x2 r724, r715, r721;
}
{
sub.f16x2 r727, %74, %63;
}
{
mul.f16x2 r730, r727, r642;
}
{
sub.f16x2 r733, %91, %73;
}
{
mul.f16x2 r736, r733, r644;
}
{
add.f16x2 r739, r730, r736;
}
{
add.f16x2 r742, r724, r739;
}
{
add.f16x2 r745, %99, %85;
}
{
mul.f16x2 r748, r745, r643;
}
{
add.f16x2 r751, %56, r748;
}
{
add.f16x2 r754, %64, %98;
}
{
mul.f16x2 r757, r754, r645;
}
{
add.f16x2 r760, r751, r757;
}
{
sub.f16x2 r763, %74, %63;
}
{
mul.f16x2 r766, r763, r644;
}
{
sub.f16x2 r769, %91, %73;
}
{
mul.f16x2 r772, r769, r647;
}
{
add.f16x2 r775, r766, r772;
}
{
sub.f16x2 r778, r760, r775;
}
{
add.f16x2 r781, %99, %85;
}
{
mul.f16x2 r784, r781, r643;
}
{
add.f16x2 r787, %56, r784;
}
{
add.f16x2 r790, %64, %98;
}
{
mul.f16x2 r793, r790, r645;
}
{
add.f16x2 r796, r787, r793;
}
{
sub.f16x2 r799, %74, %63;
}
{
mul.f16x2 r802, r799, r644;
}
{
sub.f16x2 r805, %91, %73;
}
{
mul.f16x2 r808, r805, r647;
}
{
add.f16x2 r811, r802, r808;
}
{
add.f16x2 r814, r796, r811;
}
{
add.f16x2 r817, %74, %63;
}
{
mul.f16x2 r820, r817, r641;
}
{
add.f16x2 r823, %82, r820;
}
{
add.f16x2 r826, %91, %73;
}
{
mul.f16x2 r829, r826, r643;
}
{
add.f16x2 r832, r823, r829;
}
{
sub.f16x2 r835, %99, %85;
}
{
mul.f16x2 r838, r835, r642;
}
{
sub.f16x2 r841, %64, %98;
}
{
mul.f16x2 r844, r841, r644;
}
{
add.f16x2 r847, r838, r844;
}
{
add.f16x2 r850, r832, r847;
}
{
add.f16x2 r853, %74, %63;
}
{
mul.f16x2 r856, r853, r641;
}
{
add.f16x2 r859, %82, r856;
}
{
add.f16x2 r862, %91, %73;
}
{
mul.f16x2 r865, r862, r643;
}
{
add.f16x2 r868, r859, r865;
}
{
sub.f16x2 r871, %99, %85;
}
{
mul.f16x2 r874, r871, r642;
}
{
sub.f16x2 r877, %64, %98;
}
{
mul.f16x2 r880, r877, r644;
}
{
add.f16x2 r883, r874, r880;
}
{
sub.f16x2 r886, r868, r883;
}
{
add.f16x2 r889, %74, %63;
}
{
mul.f16x2 r892, r889, r643;
}
{
add.f16x2 r895, %82, r892;
}
{
add.f16x2 r898, %91, %73;
}
{
mul.f16x2 r901, r898, r645;
}
{
add.f16x2 r904, r895, r901;
}
{
sub.f16x2 r907, %99, %85;
}
{
mul.f16x2 r910, r907, r644;
}
{
sub.f16x2 r913, %64, %98;
}
{
mul.f16x2 r916, r913, r647;
}
{
add.f16x2 r919, r910, r916;
}
{
add.f16x2 r922, r904, r919;
}
{
add.f16x2 r925, %74, %63;
}
{
mul.f16x2 r928, r925, r643;
}
{
add.f16x2 r931, %82, r928;
}
{
add.f16x2 r934, %91, %73;
}
{
mul.f16x2 r937, r934, r645;
}
{
add.f16x2 r940, r931, r937;
}
{
sub.f16x2 r943, %99, %85;
}
{
mul.f16x2 r946, r943, r644;
}
{
sub.f16x2 r949, %64, %98;
}
{
mul.f16x2 r952, r949, r647;
}
{
add.f16x2 r955, r946, r952;
}
{
sub.f16x2 r958, r940, r955;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r961, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r962, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r965, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r966, {low, high};
}
{
neg.f16x2 r967, r966;
}
{
add.f16x2 r969, %51, %88;
}
{
add.f16x2 r972, %57, r969;
}
{
add.f16x2 r975, %66, %100;
}
{
add.f16x2 r978, r972, r975;
}
{
add.f16x2 r981, %76, %65;
}
{
add.f16x2 r984, %84, r981;
}
{
add.f16x2 r987, %92, %75;
}
{
add.f16x2 r990, r984, r987;
}
{
add.f16x2 r993, %51, %88;
}
{
mul.f16x2 r996, r993, r961;
}
{
add.f16x2 r999, %57, r996;
}
{
add.f16x2 r1002, %66, %100;
}
{
mul.f16x2 r1005, r1002, r963;
}
{
add.f16x2 r1008, r999, r1005;
}
{
sub.f16x2 r1011, %76, %65;
}
{
mul.f16x2 r1014, r1011, r962;
}
{
sub.f16x2 r1017, %92, %75;
}
{
mul.f16x2 r1020, r1017, r964;
}
{
add.f16x2 r1023, r1014, r1020;
}
{
sub.f16x2 r1026, r1008, r1023;
}
{
add.f16x2 r1029, %51, %88;
}
{
mul.f16x2 r1032, r1029, r961;
}
{
add.f16x2 r1035, %57, r1032;
}
{
add.f16x2 r1038, %66, %100;
}
{
mul.f16x2 r1041, r1038, r963;
}
{
add.f16x2 r1044, r1035, r1041;
}
{
sub.f16x2 r1047, %76, %65;
}
{
mul.f16x2 r1050, r1047, r962;
}
{
sub.f16x2 r1053, %92, %75;
}
{
mul.f16x2 r1056, r1053, r964;
}
{
add.f16x2 r1059, r1050, r1056;
}
{
add.f16x2 r1062, r1044, r1059;
}
{
add.f16x2 r1065, %51, %88;
}
{
mul.f16x2 r1068, r1065, r963;
}
{
add.f16x2 r1071, %57, r1068;
}
{
add.f16x2 r1074, %66, %100;
}
{
mul.f16x2 r1077, r1074, r965;
}
{
add.f16x2 r1080, r1071, r1077;
}
{
sub.f16x2 r1083, %76, %65;
}
{
mul.f16x2 r1086, r1083, r964;
}
{
sub.f16x2 r1089, %92, %75;
}
{
mul.f16x2 r1092, r1089, r967;
}
{
add.f16x2 r1095, r1086, r1092;
}
{
sub.f16x2 r1098, r1080, r1095;
}
{
add.f16x2 r1101, %51, %88;
}
{
mul.f16x2 r1104, r1101, r963;
}
{
add.f16x2 r1107, %57, r1104;
}
{
add.f16x2 r1110, %66, %100;
}
{
mul.f16x2 r1113, r1110, r965;
}
{
add.f16x2 r1116, r1107, r1113;
}
{
sub.f16x2 r1119, %76, %65;
}
{
mul.f16x2 r1122, r1119, r964;
}
{
sub.f16x2 r1125, %92, %75;
}
{
mul.f16x2 r1128, r1125, r967;
}
{
add.f16x2 r1131, r1122, r1128;
}
{
add.f16x2 r1134, r1116, r1131;
}
{
add.f16x2 r1137, %76, %65;
}
{
mul.f16x2 r1140, r1137, r961;
}
{
add.f16x2 r1143, %84, r1140;
}
{
add.f16x2 r1146, %92, %75;
}
{
mul.f16x2 r1149, r1146, r963;
}
{
add.f16x2 r1152, r1143, r1149;
}
{
sub.f16x2 r1155, %51, %88;
}
{
mul.f16x2 r1158, r1155, r962;
}
{
sub.f16x2 r1161, %66, %100;
}
{
mul.f16x2 r1164, r1161, r964;
}
{
add.f16x2 r1167, r1158, r1164;
}
{
add.f16x2 r1170, r1152, r1167;
}
{
add.f16x2 r1173, %76, %65;
}
{
mul.f16x2 r1176, r1173, r961;
}
{
add.f16x2 r1179, %84, r1176;
}
{
add.f16x2 r1182, %92, %75;
}
{
mul.f16x2 r1185, r1182, r963;
}
{
add.f16x2 r1188, r1179, r1185;
}
{
sub.f16x2 r1191, %51, %88;
}
{
mul.f16x2 r1194, r1191, r962;
}
{
sub.f16x2 r1197, %66, %100;
}
{
mul.f16x2 r1200, r1197, r964;
}
{
add.f16x2 r1203, r1194, r1200;
}
{
sub.f16x2 r1206, r1188, r1203;
}
{
add.f16x2 r1209, %76, %65;
}
{
mul.f16x2 r1212, r1209, r963;
}
{
add.f16x2 r1215, %84, r1212;
}
{
add.f16x2 r1218, %92, %75;
}
{
mul.f16x2 r1221, r1218, r965;
}
{
add.f16x2 r1224, r1215, r1221;
}
{
sub.f16x2 r1227, %51, %88;
}
{
mul.f16x2 r1230, r1227, r964;
}
{
sub.f16x2 r1233, %66, %100;
}
{
mul.f16x2 r1236, r1233, r967;
}
{
add.f16x2 r1239, r1230, r1236;
}
{
add.f16x2 r1242, r1224, r1239;
}
{
add.f16x2 r1245, %76, %65;
}
{
mul.f16x2 r1248, r1245, r963;
}
{
add.f16x2 r1251, %84, r1248;
}
{
add.f16x2 r1254, %92, %75;
}
{
mul.f16x2 r1257, r1254, r965;
}
{
add.f16x2 r1260, r1251, r1257;
}
{
sub.f16x2 r1263, %51, %88;
}
{
mul.f16x2 r1266, r1263, r964;
}
{
sub.f16x2 r1269, %66, %100;
}
{
mul.f16x2 r1272, r1269, r967;
}
{
add.f16x2 r1275, r1266, r1272;
}
{
sub.f16x2 r1278, r1260, r1275;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1281, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1283, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1284, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1285, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1286, {low, high};
}
{
neg.f16x2 r1287, r1286;
}
{
add.f16x2 r1289, %53, %90;
}
{
add.f16x2 r1292, %60, r1289;
}
{
add.f16x2 r1295, %68, %52;
}
{
add.f16x2 r1298, r1292, r1295;
}
{
add.f16x2 r1301, %78, %67;
}
{
add.f16x2 r1304, %86, r1301;
}
{
add.f16x2 r1307, %93, %77;
}
{
add.f16x2 r1310, r1304, r1307;
}
{
add.f16x2 r1313, %53, %90;
}
{
mul.f16x2 r1316, r1313, r1281;
}
{
add.f16x2 r1319, %60, r1316;
}
{
add.f16x2 r1322, %68, %52;
}
{
mul.f16x2 r1325, r1322, r1283;
}
{
add.f16x2 r1328, r1319, r1325;
}
{
sub.f16x2 r1331, %78, %67;
}
{
mul.f16x2 r1334, r1331, r1282;
}
{
sub.f16x2 r1337, %93, %77;
}
{
mul.f16x2 r1340, r1337, r1284;
}
{
add.f16x2 r1343, r1334, r1340;
}
{
sub.f16x2 r1346, r1328, r1343;
}
{
add.f16x2 r1349, %53, %90;
}
{
mul.f16x2 r1352, r1349, r1281;
}
{
add.f16x2 r1355, %60, r1352;
}
{
add.f16x2 r1358, %68, %52;
}
{
mul.f16x2 r1361, r1358, r1283;
}
{
add.f16x2 r1364, r1355, r1361;
}
{
sub.f16x2 r1367, %78, %67;
}
{
mul.f16x2 r1370, r1367, r1282;
}
{
sub.f16x2 r1373, %93, %77;
}
{
mul.f16x2 r1376, r1373, r1284;
}
{
add.f16x2 r1379, r1370, r1376;
}
{
add.f16x2 r1382, r1364, r1379;
}
{
add.f16x2 r1385, %53, %90;
}
{
mul.f16x2 r1388, r1385, r1283;
}
{
add.f16x2 r1391, %60, r1388;
}
{
add.f16x2 r1394, %68, %52;
}
{
mul.f16x2 r1397, r1394, r1285;
}
{
add.f16x2 r1400, r1391, r1397;
}
{
sub.f16x2 r1403, %78, %67;
}
{
mul.f16x2 r1406, r1403, r1284;
}
{
sub.f16x2 r1409, %93, %77;
}
{
mul.f16x2 r1412, r1409, r1287;
}
{
add.f16x2 r1415, r1406, r1412;
}
{
sub.f16x2 r1418, r1400, r1415;
}
{
add.f16x2 r1421, %53, %90;
}
{
mul.f16x2 r1424, r1421, r1283;
}
{
add.f16x2 r1427, %60, r1424;
}
{
add.f16x2 r1430, %68, %52;
}
{
mul.f16x2 r1433, r1430, r1285;
}
{
add.f16x2 r1436, r1427, r1433;
}
{
sub.f16x2 r1439, %78, %67;
}
{
mul.f16x2 r1442, r1439, r1284;
}
{
sub.f16x2 r1445, %93, %77;
}
{
mul.f16x2 r1448, r1445, r1287;
}
{
add.f16x2 r1451, r1442, r1448;
}
{
add.f16x2 r1454, r1436, r1451;
}
{
add.f16x2 r1457, %78, %67;
}
{
mul.f16x2 r1460, r1457, r1281;
}
{
add.f16x2 r1463, %86, r1460;
}
{
add.f16x2 r1466, %93, %77;
}
{
mul.f16x2 r1469, r1466, r1283;
}
{
add.f16x2 r1472, r1463, r1469;
}
{
sub.f16x2 r1475, %53, %90;
}
{
mul.f16x2 r1478, r1475, r1282;
}
{
sub.f16x2 r1481, %68, %52;
}
{
mul.f16x2 r1484, r1481, r1284;
}
{
add.f16x2 r1487, r1478, r1484;
}
{
add.f16x2 r1490, r1472, r1487;
}
{
add.f16x2 r1493, %78, %67;
}
{
mul.f16x2 r1496, r1493, r1281;
}
{
add.f16x2 r1499, %86, r1496;
}
{
add.f16x2 r1502, %93, %77;
}
{
mul.f16x2 r1505, r1502, r1283;
}
{
add.f16x2 r1508, r1499, r1505;
}
{
sub.f16x2 r1511, %53, %90;
}
{
mul.f16x2 r1514, r1511, r1282;
}
{
sub.f16x2 r1517, %68, %52;
}
{
mul.f16x2 r1520, r1517, r1284;
}
{
add.f16x2 r1523, r1514, r1520;
}
{
sub.f16x2 r1526, r1508, r1523;
}
{
add.f16x2 r1529, %78, %67;
}
{
mul.f16x2 r1532, r1529, r1283;
}
{
add.f16x2 r1535, %86, r1532;
}
{
add.f16x2 r1538, %93, %77;
}
{
mul.f16x2 r1541, r1538, r1285;
}
{
add.f16x2 r1544, r1535, r1541;
}
{
sub.f16x2 r1547, %53, %90;
}
{
mul.f16x2 r1550, r1547, r1284;
}
{
sub.f16x2 r1553, %68, %52;
}
{
mul.f16x2 r1556, r1553, r1287;
}
{
add.f16x2 r1559, r1550, r1556;
}
{
add.f16x2 r1562, r1544, r1559;
}
{
add.f16x2 r1565, %78, %67;
}
{
mul.f16x2 r1568, r1565, r1283;
}
{
add.f16x2 r1571, %86, r1568;
}
{
add.f16x2 r1574, %93, %77;
}
{
mul.f16x2 r1577, r1574, r1285;
}
{
add.f16x2 r1580, r1571, r1577;
}
{
sub.f16x2 r1583, %53, %90;
}
{
mul.f16x2 r1586, r1583, r1284;
}
{
sub.f16x2 r1589, %68, %52;
}
{
mul.f16x2 r1592, r1589, r1287;
}
{
add.f16x2 r1595, r1586, r1592;
}
{
sub.f16x2 r1598, r1580, r1595;
}
mov.f32 f330, 0f3F77F511;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r1601, {low, high};
}
mov.f32 f332, 0fBE7EA890;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r1602, {low, high};
}
mov.f32 f334, 0f3F6055A2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r1603, {low, high};
}
mov.f32 f336, 0fBEF6A86B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r1604, {low, high};
}
mov.f32 f338, 0f3F3A9DB0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r1605, {low, high};
}
mov.f32 f340, 0fBF2F3E7B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r1606, {low, high};
}
mov.f32 f342, 0f3F092BF2;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r1607, {low, high};
}
mov.f32 f344, 0fBF5825E0;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r1608, {low, high};
}
mov.f32 f350, 0f3D809851;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r1611, {low, high};
}
mov.f32 f352, 0fBF7F7EAE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r1612, {low, high};
}
mov.f32 f358, 0fBED9FFBE;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r1615, {low, high};
}
mov.f32 f360, 0fBF67A2BF;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r1616, {low, high};
}
mov.f32 f390, 0fBF232E38;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1617, {low, high};
}
mov.f32 f364, 0fBF45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r1618, {low, high};
}
mov.f32 f374, 0fBF7DFB3B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r1623, {low, high};
}
mov.f32 f376, 0fBE00575B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r1624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r1631, {low, high};
}
mov.f32 f392, 0f3F45405B;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r1632, {low, high};
}
{
mul.f16x2 r1649, r386, r1601;
}
{
mul.f16x2 r1652, r530, r1602;
}
{
sub.f16x2 r1655, r1649, r1652;
}
{
mul.f16x2 r1658, r386, r1602;
}
{
fma.rn.f16x2 r1661, r530, r1601, r1658;
}
{
mul.f16x2 r1665, r706, r1603;
}
{
mul.f16x2 r1668, r850, r1604;
}
{
sub.f16x2 r1671, r1665, r1668;
}
{
mul.f16x2 r1674, r706, r1604;
}
{
fma.rn.f16x2 r1677, r850, r1603, r1674;
}
{
mul.f16x2 r1681, r1026, r1605;
}
{
mul.f16x2 r1684, r1170, r1606;
}
{
sub.f16x2 r1687, r1681, r1684;
}
{
mul.f16x2 r1690, r1026, r1606;
}
{
fma.rn.f16x2 r1693, r1170, r1605, r1690;
}
{
mul.f16x2 r1697, r1346, r1607;
}
{
mul.f16x2 r1700, r1490, r1608;
}
{
sub.f16x2 r1703, r1697, r1700;
}
{
mul.f16x2 r1706, r1346, r1608;
}
{
fma.rn.f16x2 r1709, r1490, r1607, r1706;
}
{
mul.f16x2 r1713, r458, r1603;
}
{
mul.f16x2 r1716, r602, r1604;
}
{
sub.f16x2 r1719, r1713, r1716;
}
{
mul.f16x2 r1722, r458, r1604;
}
{
fma.rn.f16x2 r1725, r602, r1603, r1722;
}
{
mul.f16x2 r1729, r778, r1607;
}
{
mul.f16x2 r1732, r922, r1608;
}
{
sub.f16x2 r1735, r1729, r1732;
}
{
mul.f16x2 r1738, r778, r1608;
}
{
fma.rn.f16x2 r1741, r922, r1607, r1738;
}
{
mul.f16x2 r1745, r1098, r1611;
}
{
mul.f16x2 r1748, r1242, r1612;
}
{
sub.f16x2 r1751, r1745, r1748;
}
{
mul.f16x2 r1754, r1098, r1612;
}
{
fma.rn.f16x2 r1757, r1242, r1611, r1754;
}
{
mul.f16x2 r1761, r1418, r1615;
}
{
mul.f16x2 r1764, r1562, r1616;
}
{
sub.f16x2 r1767, r1761, r1764;
}
{
mul.f16x2 r1770, r1418, r1616;
}
{
fma.rn.f16x2 r1773, r1562, r1615, r1770;
}
{
mul.f16x2 r1777, r494, r1605;
}
{
mul.f16x2 r1780, r638, r1606;
}
{
sub.f16x2 r1783, r1777, r1780;
}
{
mul.f16x2 r1786, r494, r1606;
}
{
fma.rn.f16x2 r1789, r638, r1605, r1786;
}
{
mul.f16x2 r1793, r814, r1611;
}
{
mul.f16x2 r1796, r958, r1612;
}
{
sub.f16x2 r1799, r1793, r1796;
}
{
mul.f16x2 r1802, r814, r1612;
}
{
fma.rn.f16x2 r1805, r958, r1611, r1802;
}
{
mul.f16x2 r1809, r1134, r1617;
}
{
mul.f16x2 r1812, r1278, r1618;
}
{
sub.f16x2 r1815, r1809, r1812;
}
{
mul.f16x2 r1818, r1134, r1618;
}
{
fma.rn.f16x2 r1821, r1278, r1617, r1818;
}
{
mul.f16x2 r1825, r1454, r1623;
}
{
mul.f16x2 r1828, r1598, r1624;
}
{
sub.f16x2 r1831, r1825, r1828;
}
{
mul.f16x2 r1834, r1454, r1624;
}
{
fma.rn.f16x2 r1837, r1598, r1623, r1834;
}
{
mul.f16x2 r1841, r422, r1607;
}
{
mul.f16x2 r1844, r566, r1608;
}
{
sub.f16x2 r1847, r1841, r1844;
}
{
mul.f16x2 r1850, r422, r1608;
}
{
fma.rn.f16x2 r1853, r566, r1607, r1850;
}
{
mul.f16x2 r1857, r742, r1615;
}
{
mul.f16x2 r1860, r886, r1616;
}
{
sub.f16x2 r1863, r1857, r1860;
}
{
mul.f16x2 r1866, r742, r1616;
}
{
fma.rn.f16x2 r1869, r886, r1615, r1866;
}
{
mul.f16x2 r1873, r1062, r1623;
}
{
mul.f16x2 r1876, r1206, r1624;
}
{
sub.f16x2 r1879, r1873, r1876;
}
{
mul.f16x2 r1882, r1062, r1624;
}
{
fma.rn.f16x2 r1885, r1206, r1623, r1882;
}
{
mul.f16x2 r1889, r1382, r1631;
}
{
mul.f16x2 r1892, r1526, r1632;
}
{
sub.f16x2 r1895, r1889, r1892;
}
{
mul.f16x2 r1898, r1382, r1632;
}
{
fma.rn.f16x2 r1901, r1526, r1631, r1898;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1905, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1906, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r1907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r1908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r1909, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r1910, {low, high};
}
{
neg.f16x2 r1911, r1910;
}
{
add.f16x2 r1913, r338, r1298;
}
{
add.f16x2 r1916, r18, r1913;
}
{
add.f16x2 r1919, r658, r978;
}
{
add.f16x2 r1922, r1916, r1919;
}
{
add.f16x2 r1925, r350, r1310;
}
{
add.f16x2 r1928, r30, r1925;
}
{
add.f16x2 r1931, r670, r990;
}
{
add.f16x2 r1934, r1928, r1931;
}
{
add.f16x2 r1937, r338, r1298;
}
{
mul.f16x2 r1940, r1937, r1905;
}
{
add.f16x2 r1943, r18, r1940;
}
{
add.f16x2 r1946, r658, r978;
}
{
mul.f16x2 r1949, r1946, r1907;
}
{
add.f16x2 r1952, r1943, r1949;
}
{
sub.f16x2 r1955, r350, r1310;
}
{
mul.f16x2 r1958, r1955, r1906;
}
{
sub.f16x2 r1961, r670, r990;
}
{
mul.f16x2 r1964, r1961, r1908;
}
{
add.f16x2 r1967, r1958, r1964;
}
{
sub.f16x2 r1970, r1952, r1967;
}
{
add.f16x2 r1973, r338, r1298;
}
{
mul.f16x2 r1976, r1973, r1905;
}
{
add.f16x2 r1979, r18, r1976;
}
{
add.f16x2 r1982, r658, r978;
}
{
mul.f16x2 r1985, r1982, r1907;
}
{
add.f16x2 r1988, r1979, r1985;
}
{
sub.f16x2 r1991, r350, r1310;
}
{
mul.f16x2 r1994, r1991, r1906;
}
{
sub.f16x2 r1997, r670, r990;
}
{
mul.f16x2 r2000, r1997, r1908;
}
{
add.f16x2 r2003, r1994, r2000;
}
{
add.f16x2 r2006, r1988, r2003;
}
{
add.f16x2 r2009, r338, r1298;
}
{
mul.f16x2 r2012, r2009, r1907;
}
{
add.f16x2 r2015, r18, r2012;
}
{
add.f16x2 r2018, r658, r978;
}
{
mul.f16x2 r2021, r2018, r1909;
}
{
add.f16x2 r2024, r2015, r2021;
}
{
sub.f16x2 r2027, r350, r1310;
}
{
mul.f16x2 r2030, r2027, r1908;
}
{
sub.f16x2 r2033, r670, r990;
}
{
mul.f16x2 r2036, r2033, r1911;
}
{
add.f16x2 r2039, r2030, r2036;
}
{
sub.f16x2 r2042, r2024, r2039;
}
{
add.f16x2 r2045, r338, r1298;
}
{
mul.f16x2 r2048, r2045, r1907;
}
{
add.f16x2 r2051, r18, r2048;
}
{
add.f16x2 r2054, r658, r978;
}
{
mul.f16x2 r2057, r2054, r1909;
}
{
add.f16x2 r2060, r2051, r2057;
}
{
sub.f16x2 r2063, r350, r1310;
}
{
mul.f16x2 r2066, r2063, r1908;
}
{
sub.f16x2 r2069, r670, r990;
}
{
mul.f16x2 r2072, r2069, r1911;
}
{
add.f16x2 r2075, r2066, r2072;
}
{
add.f16x2 r2078, r2060, r2075;
}
{
add.f16x2 r2081, r350, r1310;
}
{
mul.f16x2 r2084, r2081, r1905;
}
{
add.f16x2 r2087, r30, r2084;
}
{
add.f16x2 r2090, r670, r990;
}
{
mul.f16x2 r2093, r2090, r1907;
}
{
add.f16x2 r2096, r2087, r2093;
}
{
sub.f16x2 r2099, r338, r1298;
}
{
mul.f16x2 r2102, r2099, r1906;
}
{
sub.f16x2 r2105, r658, r978;
}
{
mul.f16x2 r2108, r2105, r1908;
}
{
add.f16x2 r2111, r2102, r2108;
}
{
add.f16x2 r2114, r2096, r2111;
}
{
add.f16x2 r2117, r350, r1310;
}
{
mul.f16x2 r2120, r2117, r1905;
}
{
add.f16x2 r2123, r30, r2120;
}
{
add.f16x2 r2126, r670, r990;
}
{
mul.f16x2 r2129, r2126, r1907;
}
{
add.f16x2 r2132, r2123, r2129;
}
{
sub.f16x2 r2135, r338, r1298;
}
{
mul.f16x2 r2138, r2135, r1906;
}
{
sub.f16x2 r2141, r658, r978;
}
{
mul.f16x2 r2144, r2141, r1908;
}
{
add.f16x2 r2147, r2138, r2144;
}
{
sub.f16x2 r2150, r2132, r2147;
}
{
add.f16x2 r2153, r350, r1310;
}
{
mul.f16x2 r2156, r2153, r1907;
}
{
add.f16x2 r2159, r30, r2156;
}
{
add.f16x2 r2162, r670, r990;
}
{
mul.f16x2 r2165, r2162, r1909;
}
{
add.f16x2 r2168, r2159, r2165;
}
{
sub.f16x2 r2171, r338, r1298;
}
{
mul.f16x2 r2174, r2171, r1908;
}
{
sub.f16x2 r2177, r658, r978;
}
{
mul.f16x2 r2180, r2177, r1911;
}
{
add.f16x2 r2183, r2174, r2180;
}
{
add.f16x2 r2186, r2168, r2183;
}
{
add.f16x2 r2189, r350, r1310;
}
{
mul.f16x2 r2192, r2189, r1907;
}
{
add.f16x2 r2195, r30, r2192;
}
{
add.f16x2 r2198, r670, r990;
}
{
mul.f16x2 r2201, r2198, r1909;
}
{
add.f16x2 r2204, r2195, r2201;
}
{
sub.f16x2 r2207, r338, r1298;
}
{
mul.f16x2 r2210, r2207, r1908;
}
{
sub.f16x2 r2213, r658, r978;
}
{
mul.f16x2 r2216, r2213, r1911;
}
{
add.f16x2 r2219, r2210, r2216;
}
{
sub.f16x2 r2222, r2204, r2219;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2226, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2227, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2228, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2230, {low, high};
}
{
neg.f16x2 r2231, r2230;
}
{
add.f16x2 r2233, r1655, r1703;
}
{
add.f16x2 r2236, r66, r2233;
}
{
add.f16x2 r2239, r1671, r1687;
}
{
add.f16x2 r2242, r2236, r2239;
}
{
add.f16x2 r2245, r1661, r1709;
}
{
add.f16x2 r2248, r210, r2245;
}
{
add.f16x2 r2251, r1677, r1693;
}
{
add.f16x2 r2254, r2248, r2251;
}
{
add.f16x2 r2257, r1655, r1703;
}
{
mul.f16x2 r2260, r2257, r2225;
}
{
add.f16x2 r2263, r66, r2260;
}
{
add.f16x2 r2266, r1671, r1687;
}
{
mul.f16x2 r2269, r2266, r2227;
}
{
add.f16x2 r2272, r2263, r2269;
}
{
sub.f16x2 r2275, r1661, r1709;
}
{
mul.f16x2 r2278, r2275, r2226;
}
{
sub.f16x2 r2281, r1677, r1693;
}
{
mul.f16x2 r2284, r2281, r2228;
}
{
add.f16x2 r2287, r2278, r2284;
}
{
sub.f16x2 r2290, r2272, r2287;
}
{
add.f16x2 r2293, r1655, r1703;
}
{
mul.f16x2 r2296, r2293, r2225;
}
{
add.f16x2 r2299, r66, r2296;
}
{
add.f16x2 r2302, r1671, r1687;
}
{
mul.f16x2 r2305, r2302, r2227;
}
{
add.f16x2 r2308, r2299, r2305;
}
{
sub.f16x2 r2311, r1661, r1709;
}
{
mul.f16x2 r2314, r2311, r2226;
}
{
sub.f16x2 r2317, r1677, r1693;
}
{
mul.f16x2 r2320, r2317, r2228;
}
{
add.f16x2 r2323, r2314, r2320;
}
{
add.f16x2 r2326, r2308, r2323;
}
{
add.f16x2 r2329, r1655, r1703;
}
{
mul.f16x2 r2332, r2329, r2227;
}
{
add.f16x2 r2335, r66, r2332;
}
{
add.f16x2 r2338, r1671, r1687;
}
{
mul.f16x2 r2341, r2338, r2229;
}
{
add.f16x2 r2344, r2335, r2341;
}
{
sub.f16x2 r2347, r1661, r1709;
}
{
mul.f16x2 r2350, r2347, r2228;
}
{
sub.f16x2 r2353, r1677, r1693;
}
{
mul.f16x2 r2356, r2353, r2231;
}
{
add.f16x2 r2359, r2350, r2356;
}
{
sub.f16x2 r2362, r2344, r2359;
}
{
add.f16x2 r2365, r1655, r1703;
}
{
mul.f16x2 r2368, r2365, r2227;
}
{
add.f16x2 r2371, r66, r2368;
}
{
add.f16x2 r2374, r1671, r1687;
}
{
mul.f16x2 r2377, r2374, r2229;
}
{
add.f16x2 r2380, r2371, r2377;
}
{
sub.f16x2 r2383, r1661, r1709;
}
{
mul.f16x2 r2386, r2383, r2228;
}
{
sub.f16x2 r2389, r1677, r1693;
}
{
mul.f16x2 r2392, r2389, r2231;
}
{
add.f16x2 r2395, r2386, r2392;
}
{
add.f16x2 r2398, r2380, r2395;
}
{
add.f16x2 r2401, r1661, r1709;
}
{
mul.f16x2 r2404, r2401, r2225;
}
{
add.f16x2 r2407, r210, r2404;
}
{
add.f16x2 r2410, r1677, r1693;
}
{
mul.f16x2 r2413, r2410, r2227;
}
{
add.f16x2 r2416, r2407, r2413;
}
{
sub.f16x2 r2419, r1655, r1703;
}
{
mul.f16x2 r2422, r2419, r2226;
}
{
sub.f16x2 r2425, r1671, r1687;
}
{
mul.f16x2 r2428, r2425, r2228;
}
{
add.f16x2 r2431, r2422, r2428;
}
{
add.f16x2 r2434, r2416, r2431;
}
{
add.f16x2 r2437, r1661, r1709;
}
{
mul.f16x2 r2440, r2437, r2225;
}
{
add.f16x2 r2443, r210, r2440;
}
{
add.f16x2 r2446, r1677, r1693;
}
{
mul.f16x2 r2449, r2446, r2227;
}
{
add.f16x2 r2452, r2443, r2449;
}
{
sub.f16x2 r2455, r1655, r1703;
}
{
mul.f16x2 r2458, r2455, r2226;
}
{
sub.f16x2 r2461, r1671, r1687;
}
{
mul.f16x2 r2464, r2461, r2228;
}
{
add.f16x2 r2467, r2458, r2464;
}
{
sub.f16x2 r2470, r2452, r2467;
}
{
add.f16x2 r2473, r1661, r1709;
}
{
mul.f16x2 r2476, r2473, r2227;
}
{
add.f16x2 r2479, r210, r2476;
}
{
add.f16x2 r2482, r1677, r1693;
}
{
mul.f16x2 r2485, r2482, r2229;
}
{
add.f16x2 r2488, r2479, r2485;
}
{
sub.f16x2 r2491, r1655, r1703;
}
{
mul.f16x2 r2494, r2491, r2228;
}
{
sub.f16x2 r2497, r1671, r1687;
}
{
mul.f16x2 r2500, r2497, r2231;
}
{
add.f16x2 r2503, r2494, r2500;
}
{
add.f16x2 r2506, r2488, r2503;
}
{
add.f16x2 r2509, r1661, r1709;
}
{
mul.f16x2 r2512, r2509, r2227;
}
{
add.f16x2 r2515, r210, r2512;
}
{
add.f16x2 r2518, r1677, r1693;
}
{
mul.f16x2 r2521, r2518, r2229;
}
{
add.f16x2 r2524, r2515, r2521;
}
{
sub.f16x2 r2527, r1655, r1703;
}
{
mul.f16x2 r2530, r2527, r2228;
}
{
sub.f16x2 r2533, r1671, r1687;
}
{
mul.f16x2 r2536, r2533, r2231;
}
{
add.f16x2 r2539, r2530, r2536;
}
{
sub.f16x2 r2542, r2524, r2539;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2545, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2546, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2547, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2548, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2549, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2550, {low, high};
}
{
neg.f16x2 r2551, r2550;
}
{
add.f16x2 r2553, r1719, r1767;
}
{
add.f16x2 r2556, r138, r2553;
}
{
add.f16x2 r2559, r1735, r1751;
}
{
add.f16x2 r2562, r2556, r2559;
}
{
add.f16x2 r2565, r1725, r1773;
}
{
add.f16x2 r2568, r282, r2565;
}
{
add.f16x2 r2571, r1741, r1757;
}
{
add.f16x2 r2574, r2568, r2571;
}
{
add.f16x2 r2577, r1719, r1767;
}
{
mul.f16x2 r2580, r2577, r2545;
}
{
add.f16x2 r2583, r138, r2580;
}
{
add.f16x2 r2586, r1735, r1751;
}
{
mul.f16x2 r2589, r2586, r2547;
}
{
add.f16x2 r2592, r2583, r2589;
}
{
sub.f16x2 r2595, r1725, r1773;
}
{
mul.f16x2 r2598, r2595, r2546;
}
{
sub.f16x2 r2601, r1741, r1757;
}
{
mul.f16x2 r2604, r2601, r2548;
}
{
add.f16x2 r2607, r2598, r2604;
}
{
sub.f16x2 r2610, r2592, r2607;
}
{
add.f16x2 r2613, r1719, r1767;
}
{
mul.f16x2 r2616, r2613, r2545;
}
{
add.f16x2 r2619, r138, r2616;
}
{
add.f16x2 r2622, r1735, r1751;
}
{
mul.f16x2 r2625, r2622, r2547;
}
{
add.f16x2 r2628, r2619, r2625;
}
{
sub.f16x2 r2631, r1725, r1773;
}
{
mul.f16x2 r2634, r2631, r2546;
}
{
sub.f16x2 r2637, r1741, r1757;
}
{
mul.f16x2 r2640, r2637, r2548;
}
{
add.f16x2 r2643, r2634, r2640;
}
{
add.f16x2 r2646, r2628, r2643;
}
{
add.f16x2 r2649, r1719, r1767;
}
{
mul.f16x2 r2652, r2649, r2547;
}
{
add.f16x2 r2655, r138, r2652;
}
{
add.f16x2 r2658, r1735, r1751;
}
{
mul.f16x2 r2661, r2658, r2549;
}
{
add.f16x2 r2664, r2655, r2661;
}
{
sub.f16x2 r2667, r1725, r1773;
}
{
mul.f16x2 r2670, r2667, r2548;
}
{
sub.f16x2 r2673, r1741, r1757;
}
{
mul.f16x2 r2676, r2673, r2551;
}
{
add.f16x2 r2679, r2670, r2676;
}
{
sub.f16x2 r2682, r2664, r2679;
}
{
add.f16x2 r2685, r1719, r1767;
}
{
mul.f16x2 r2688, r2685, r2547;
}
{
add.f16x2 r2691, r138, r2688;
}
{
add.f16x2 r2694, r1735, r1751;
}
{
mul.f16x2 r2697, r2694, r2549;
}
{
add.f16x2 r2700, r2691, r2697;
}
{
sub.f16x2 r2703, r1725, r1773;
}
{
mul.f16x2 r2706, r2703, r2548;
}
{
sub.f16x2 r2709, r1741, r1757;
}
{
mul.f16x2 r2712, r2709, r2551;
}
{
add.f16x2 r2715, r2706, r2712;
}
{
add.f16x2 r2718, r2700, r2715;
}
{
add.f16x2 r2721, r1725, r1773;
}
{
mul.f16x2 r2724, r2721, r2545;
}
{
add.f16x2 r2727, r282, r2724;
}
{
add.f16x2 r2730, r1741, r1757;
}
{
mul.f16x2 r2733, r2730, r2547;
}
{
add.f16x2 r2736, r2727, r2733;
}
{
sub.f16x2 r2739, r1719, r1767;
}
{
mul.f16x2 r2742, r2739, r2546;
}
{
sub.f16x2 r2745, r1735, r1751;
}
{
mul.f16x2 r2748, r2745, r2548;
}
{
add.f16x2 r2751, r2742, r2748;
}
{
add.f16x2 r2754, r2736, r2751;
}
{
add.f16x2 r2757, r1725, r1773;
}
{
mul.f16x2 r2760, r2757, r2545;
}
{
add.f16x2 r2763, r282, r2760;
}
{
add.f16x2 r2766, r1741, r1757;
}
{
mul.f16x2 r2769, r2766, r2547;
}
{
add.f16x2 r2772, r2763, r2769;
}
{
sub.f16x2 r2775, r1719, r1767;
}
{
mul.f16x2 r2778, r2775, r2546;
}
{
sub.f16x2 r2781, r1735, r1751;
}
{
mul.f16x2 r2784, r2781, r2548;
}
{
add.f16x2 r2787, r2778, r2784;
}
{
sub.f16x2 r2790, r2772, r2787;
}
{
add.f16x2 r2793, r1725, r1773;
}
{
mul.f16x2 r2796, r2793, r2547;
}
{
add.f16x2 r2799, r282, r2796;
}
{
add.f16x2 r2802, r1741, r1757;
}
{
mul.f16x2 r2805, r2802, r2549;
}
{
add.f16x2 r2808, r2799, r2805;
}
{
sub.f16x2 r2811, r1719, r1767;
}
{
mul.f16x2 r2814, r2811, r2548;
}
{
sub.f16x2 r2817, r1735, r1751;
}
{
mul.f16x2 r2820, r2817, r2551;
}
{
add.f16x2 r2823, r2814, r2820;
}
{
add.f16x2 r2826, r2808, r2823;
}
{
add.f16x2 r2829, r1725, r1773;
}
{
mul.f16x2 r2832, r2829, r2547;
}
{
add.f16x2 r2835, r282, r2832;
}
{
add.f16x2 r2838, r1741, r1757;
}
{
mul.f16x2 r2841, r2838, r2549;
}
{
add.f16x2 r2844, r2835, r2841;
}
{
sub.f16x2 r2847, r1719, r1767;
}
{
mul.f16x2 r2850, r2847, r2548;
}
{
sub.f16x2 r2853, r1735, r1751;
}
{
mul.f16x2 r2856, r2853, r2551;
}
{
add.f16x2 r2859, r2850, r2856;
}
{
sub.f16x2 r2862, r2844, r2859;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2866, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r2867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r2868, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r2869, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r2870, {low, high};
}
{
neg.f16x2 r2871, r2870;
}
{
add.f16x2 r2873, r1783, r1831;
}
{
add.f16x2 r2876, r174, r2873;
}
{
add.f16x2 r2879, r1799, r1815;
}
{
add.f16x2 r2882, r2876, r2879;
}
{
add.f16x2 r2885, r1789, r1837;
}
{
add.f16x2 r2888, r318, r2885;
}
{
add.f16x2 r2891, r1805, r1821;
}
{
add.f16x2 r2894, r2888, r2891;
}
{
add.f16x2 r2897, r1783, r1831;
}
{
mul.f16x2 r2900, r2897, r2865;
}
{
add.f16x2 r2903, r174, r2900;
}
{
add.f16x2 r2906, r1799, r1815;
}
{
mul.f16x2 r2909, r2906, r2867;
}
{
add.f16x2 r2912, r2903, r2909;
}
{
sub.f16x2 r2915, r1789, r1837;
}
{
mul.f16x2 r2918, r2915, r2866;
}
{
sub.f16x2 r2921, r1805, r1821;
}
{
mul.f16x2 r2924, r2921, r2868;
}
{
add.f16x2 r2927, r2918, r2924;
}
{
sub.f16x2 r2930, r2912, r2927;
}
{
add.f16x2 r2933, r1783, r1831;
}
{
mul.f16x2 r2936, r2933, r2865;
}
{
add.f16x2 r2939, r174, r2936;
}
{
add.f16x2 r2942, r1799, r1815;
}
{
mul.f16x2 r2945, r2942, r2867;
}
{
add.f16x2 r2948, r2939, r2945;
}
{
sub.f16x2 r2951, r1789, r1837;
}
{
mul.f16x2 r2954, r2951, r2866;
}
{
sub.f16x2 r2957, r1805, r1821;
}
{
mul.f16x2 r2960, r2957, r2868;
}
{
add.f16x2 r2963, r2954, r2960;
}
{
add.f16x2 r2966, r2948, r2963;
}
{
add.f16x2 r2969, r1783, r1831;
}
{
mul.f16x2 r2972, r2969, r2867;
}
{
add.f16x2 r2975, r174, r2972;
}
{
add.f16x2 r2978, r1799, r1815;
}
{
mul.f16x2 r2981, r2978, r2869;
}
{
add.f16x2 r2984, r2975, r2981;
}
{
sub.f16x2 r2987, r1789, r1837;
}
{
mul.f16x2 r2990, r2987, r2868;
}
{
sub.f16x2 r2993, r1805, r1821;
}
{
mul.f16x2 r2996, r2993, r2871;
}
{
add.f16x2 r2999, r2990, r2996;
}
{
sub.f16x2 r3002, r2984, r2999;
}
{
add.f16x2 r3005, r1783, r1831;
}
{
mul.f16x2 r3008, r3005, r2867;
}
{
add.f16x2 r3011, r174, r3008;
}
{
add.f16x2 r3014, r1799, r1815;
}
{
mul.f16x2 r3017, r3014, r2869;
}
{
add.f16x2 r3020, r3011, r3017;
}
{
sub.f16x2 r3023, r1789, r1837;
}
{
mul.f16x2 r3026, r3023, r2868;
}
{
sub.f16x2 r3029, r1805, r1821;
}
{
mul.f16x2 r3032, r3029, r2871;
}
{
add.f16x2 r3035, r3026, r3032;
}
{
add.f16x2 r3038, r3020, r3035;
}
{
add.f16x2 r3041, r1789, r1837;
}
{
mul.f16x2 r3044, r3041, r2865;
}
{
add.f16x2 r3047, r318, r3044;
}
{
add.f16x2 r3050, r1805, r1821;
}
{
mul.f16x2 r3053, r3050, r2867;
}
{
add.f16x2 r3056, r3047, r3053;
}
{
sub.f16x2 r3059, r1783, r1831;
}
{
mul.f16x2 r3062, r3059, r2866;
}
{
sub.f16x2 r3065, r1799, r1815;
}
{
mul.f16x2 r3068, r3065, r2868;
}
{
add.f16x2 r3071, r3062, r3068;
}
{
add.f16x2 r3074, r3056, r3071;
}
{
add.f16x2 r3077, r1789, r1837;
}
{
mul.f16x2 r3080, r3077, r2865;
}
{
add.f16x2 r3083, r318, r3080;
}
{
add.f16x2 r3086, r1805, r1821;
}
{
mul.f16x2 r3089, r3086, r2867;
}
{
add.f16x2 r3092, r3083, r3089;
}
{
sub.f16x2 r3095, r1783, r1831;
}
{
mul.f16x2 r3098, r3095, r2866;
}
{
sub.f16x2 r3101, r1799, r1815;
}
{
mul.f16x2 r3104, r3101, r2868;
}
{
add.f16x2 r3107, r3098, r3104;
}
{
sub.f16x2 r3110, r3092, r3107;
}
{
add.f16x2 r3113, r1789, r1837;
}
{
mul.f16x2 r3116, r3113, r2867;
}
{
add.f16x2 r3119, r318, r3116;
}
{
add.f16x2 r3122, r1805, r1821;
}
{
mul.f16x2 r3125, r3122, r2869;
}
{
add.f16x2 r3128, r3119, r3125;
}
{
sub.f16x2 r3131, r1783, r1831;
}
{
mul.f16x2 r3134, r3131, r2868;
}
{
sub.f16x2 r3137, r1799, r1815;
}
{
mul.f16x2 r3140, r3137, r2871;
}
{
add.f16x2 r3143, r3134, r3140;
}
{
add.f16x2 r3146, r3128, r3143;
}
{
add.f16x2 r3149, r1789, r1837;
}
{
mul.f16x2 r3152, r3149, r2867;
}
{
add.f16x2 r3155, r318, r3152;
}
{
add.f16x2 r3158, r1805, r1821;
}
{
mul.f16x2 r3161, r3158, r2869;
}
{
add.f16x2 r3164, r3155, r3161;
}
{
sub.f16x2 r3167, r1783, r1831;
}
{
mul.f16x2 r3170, r3167, r2868;
}
{
sub.f16x2 r3173, r1799, r1815;
}
{
mul.f16x2 r3176, r3173, r2871;
}
{
add.f16x2 r3179, r3170, r3176;
}
{
sub.f16x2 r3182, r3164, r3179;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3185, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3186, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r3187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r3188, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r3189, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r3190, {low, high};
}
{
neg.f16x2 r3191, r3190;
}
{
add.f16x2 r3193, r1847, r1895;
}
{
add.f16x2 r3196, r102, r3193;
}
{
add.f16x2 r3199, r1863, r1879;
}
{
add.f16x2 r3202, r3196, r3199;
}
{
add.f16x2 r3205, r1853, r1901;
}
{
add.f16x2 r3208, r246, r3205;
}
{
add.f16x2 r3211, r1869, r1885;
}
{
add.f16x2 r3214, r3208, r3211;
}
{
add.f16x2 r3217, r1847, r1895;
}
{
mul.f16x2 r3220, r3217, r3185;
}
{
add.f16x2 r3223, r102, r3220;
}
{
add.f16x2 r3226, r1863, r1879;
}
{
mul.f16x2 r3229, r3226, r3187;
}
{
add.f16x2 r3232, r3223, r3229;
}
{
sub.f16x2 r3235, r1853, r1901;
}
{
mul.f16x2 r3238, r3235, r3186;
}
{
sub.f16x2 r3241, r1869, r1885;
}
{
mul.f16x2 r3244, r3241, r3188;
}
{
add.f16x2 r3247, r3238, r3244;
}
{
sub.f16x2 r3250, r3232, r3247;
}
{
add.f16x2 r3253, r1847, r1895;
}
{
mul.f16x2 r3256, r3253, r3185;
}
{
add.f16x2 r3259, r102, r3256;
}
{
add.f16x2 r3262, r1863, r1879;
}
{
mul.f16x2 r3265, r3262, r3187;
}
{
add.f16x2 r3268, r3259, r3265;
}
{
sub.f16x2 r3271, r1853, r1901;
}
{
mul.f16x2 r3274, r3271, r3186;
}
{
sub.f16x2 r3277, r1869, r1885;
}
{
mul.f16x2 r3280, r3277, r3188;
}
{
add.f16x2 r3283, r3274, r3280;
}
{
add.f16x2 r3286, r3268, r3283;
}
{
add.f16x2 r3289, r1847, r1895;
}
{
mul.f16x2 r3292, r3289, r3187;
}
{
add.f16x2 r3295, r102, r3292;
}
{
add.f16x2 r3298, r1863, r1879;
}
{
mul.f16x2 r3301, r3298, r3189;
}
{
add.f16x2 r3304, r3295, r3301;
}
{
sub.f16x2 r3307, r1853, r1901;
}
{
mul.f16x2 r3310, r3307, r3188;
}
{
sub.f16x2 r3313, r1869, r1885;
}
{
mul.f16x2 r3316, r3313, r3191;
}
{
add.f16x2 r3319, r3310, r3316;
}
{
sub.f16x2 r3322, r3304, r3319;
}
{
add.f16x2 r3325, r1847, r1895;
}
{
mul.f16x2 r3328, r3325, r3187;
}
{
add.f16x2 r3331, r102, r3328;
}
{
add.f16x2 r3334, r1863, r1879;
}
{
mul.f16x2 r3337, r3334, r3189;
}
{
add.f16x2 r3340, r3331, r3337;
}
{
sub.f16x2 r3343, r1853, r1901;
}
{
mul.f16x2 r3346, r3343, r3188;
}
{
sub.f16x2 r3349, r1869, r1885;
}
{
mul.f16x2 r3352, r3349, r3191;
}
{
add.f16x2 r3355, r3346, r3352;
}
{
add.f16x2 r3358, r3340, r3355;
}
{
add.f16x2 r3361, r1853, r1901;
}
{
mul.f16x2 r3364, r3361, r3185;
}
{
add.f16x2 r3367, r246, r3364;
}
{
add.f16x2 r3370, r1869, r1885;
}
{
mul.f16x2 r3373, r3370, r3187;
}
{
add.f16x2 r3376, r3367, r3373;
}
{
sub.f16x2 r3379, r1847, r1895;
}
{
mul.f16x2 r3382, r3379, r3186;
}
{
sub.f16x2 r3385, r1863, r1879;
}
{
mul.f16x2 r3388, r3385, r3188;
}
{
add.f16x2 r3391, r3382, r3388;
}
{
add.f16x2 r3394, r3376, r3391;
}
{
add.f16x2 r3397, r1853, r1901;
}
{
mul.f16x2 r3400, r3397, r3185;
}
{
add.f16x2 r3403, r246, r3400;
}
{
add.f16x2 r3406, r1869, r1885;
}
{
mul.f16x2 r3409, r3406, r3187;
}
{
add.f16x2 r3412, r3403, r3409;
}
{
sub.f16x2 r3415, r1847, r1895;
}
{
mul.f16x2 r3418, r3415, r3186;
}
{
sub.f16x2 r3421, r1863, r1879;
}
{
mul.f16x2 r3424, r3421, r3188;
}
{
add.f16x2 r3427, r3418, r3424;
}
{
sub.f16x2 r3430, r3412, r3427;
}
{
add.f16x2 r3433, r1853, r1901;
}
{
mul.f16x2 r3436, r3433, r3187;
}
{
add.f16x2 r3439, r246, r3436;
}
{
add.f16x2 r3442, r1869, r1885;
}
{
mul.f16x2 r3445, r3442, r3189;
}
{
add.f16x2 r3448, r3439, r3445;
}
{
sub.f16x2 r3451, r1847, r1895;
}
{
mul.f16x2 r3454, r3451, r3188;
}
{
sub.f16x2 r3457, r1863, r1879;
}
{
mul.f16x2 r3460, r3457, r3191;
}
{
add.f16x2 r3463, r3454, r3460;
}
{
add.f16x2 r3466, r3448, r3463;
}
{
add.f16x2 r3469, r1853, r1901;
}
{
mul.f16x2 r3472, r3469, r3187;
}
{
add.f16x2 r3475, r246, r3472;
}
{
add.f16x2 r3478, r1869, r1885;
}
{
mul.f16x2 r3481, r3478, r3189;
}
{
add.f16x2 r3484, r3475, r3481;
}
{
sub.f16x2 r3487, r1847, r1895;
}
{
mul.f16x2 r3490, r3487, r3188;
}
{
sub.f16x2 r3493, r1863, r1879;
}
{
mul.f16x2 r3496, r3493, r3191;
}
{
add.f16x2 r3499, r3490, r3496;
}
{
sub.f16x2 r3502, r3484, r3499;
}
mul.wide.u32 rd2, r10394, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r10395, rd3;
mul.lo.s32 r10396, r10395, 125;
sub.s32 r10397, r10394, r10396;
mad.lo.s32 r10398, r10395, 12500, r10393;
cvt.rn.f32.u32 f597, r10397;
mul.f32 f598, f597, 0f3B03C498;
cos.approx.f32 f217, f598;
sin.approx.f32 f599, f598;
neg.f32 f218, f599;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f217;
cvt.rn.f16.f32 high, f218;
mov.b32 r3505, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3508, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3510, {high, high};
}
{
mul.f16x2 r3512, r2254, r3510;
}
{
neg.f16x2 r3515, r3512;
}
{
fma.rn.f16x2 r3517, r2242, r3508, r3515;
}
{
mul.f16x2 r3521, r2242, r3510;
}
{
fma.rn.f16x2 r3524, r2254, r3508, r3521;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3528, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3530, {high, high};
}
mov.f32 f533, 0fBF800000;
mov.f32 f534, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3532, {low, high};
}
{
mul.f16x2 r3533, r3530, r3532;
}
{
mul.f16x2 r3536, r3505, r3528;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3539, {high, low};
}
{
fma.rn.f16x2 r3541, r3533, r3539, r3536;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3545, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3547, {high, high};
}
{
mul.f16x2 r3549, r2574, r3547;
}
{
neg.f16x2 r3552, r3549;
}
{
fma.rn.f16x2 r3554, r2562, r3545, r3552;
}
{
mul.f16x2 r3558, r2562, r3547;
}
{
fma.rn.f16x2 r3561, r2574, r3545, r3558;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3565, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3567, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3569, {low, high};
}
{
mul.f16x2 r3570, r3567, r3569;
}
{
mul.f16x2 r3573, r3541, r3565;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3541;
mov.b32 r3576, {high, low};
}
{
fma.rn.f16x2 r3578, r3570, r3576, r3573;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3582, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3584, {high, high};
}
{
mul.f16x2 r3586, r2894, r3584;
}
{
neg.f16x2 r3589, r3586;
}
{
fma.rn.f16x2 r3591, r2882, r3582, r3589;
}
{
mul.f16x2 r3595, r2882, r3584;
}
{
fma.rn.f16x2 r3598, r2894, r3582, r3595;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3602, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3604, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3606, {low, high};
}
{
mul.f16x2 r3607, r3604, r3606;
}
{
mul.f16x2 r3610, r3578, r3602;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3578;
mov.b32 r3613, {high, low};
}
{
fma.rn.f16x2 r3615, r3607, r3613, r3610;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3619, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3621, {high, high};
}
{
mul.f16x2 r3623, r3214, r3621;
}
{
neg.f16x2 r3626, r3623;
}
{
fma.rn.f16x2 r3628, r3202, r3619, r3626;
}
{
mul.f16x2 r3632, r3202, r3621;
}
{
fma.rn.f16x2 r3635, r3214, r3619, r3632;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3639, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3641, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3643, {low, high};
}
{
mul.f16x2 r3644, r3641, r3643;
}
{
mul.f16x2 r3647, r3615, r3639;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3615;
mov.b32 r3650, {high, low};
}
{
fma.rn.f16x2 r3652, r3644, r3650, r3647;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3656, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3658, {high, high};
}
{
mul.f16x2 r3660, r2114, r3658;
}
{
neg.f16x2 r3663, r3660;
}
{
fma.rn.f16x2 r3665, r1970, r3656, r3663;
}
{
mul.f16x2 r3669, r1970, r3658;
}
{
fma.rn.f16x2 r3672, r2114, r3656, r3669;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3676, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3678, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3680, {low, high};
}
{
mul.f16x2 r3681, r3678, r3680;
}
{
mul.f16x2 r3684, r3652, r3676;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3652;
mov.b32 r3687, {high, low};
}
{
fma.rn.f16x2 r3689, r3681, r3687, r3684;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3693, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3695, {high, high};
}
{
mul.f16x2 r3697, r2434, r3695;
}
{
neg.f16x2 r3700, r3697;
}
{
fma.rn.f16x2 r3702, r2290, r3693, r3700;
}
{
mul.f16x2 r3706, r2290, r3695;
}
{
fma.rn.f16x2 r3709, r2434, r3693, r3706;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3713, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3715, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3717, {low, high};
}
{
mul.f16x2 r3718, r3715, r3717;
}
{
mul.f16x2 r3721, r3689, r3713;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3689;
mov.b32 r3724, {high, low};
}
{
fma.rn.f16x2 r3726, r3718, r3724, r3721;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3730, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3732, {high, high};
}
{
mul.f16x2 r3734, r2754, r3732;
}
{
neg.f16x2 r3737, r3734;
}
{
fma.rn.f16x2 r3739, r2610, r3730, r3737;
}
{
mul.f16x2 r3743, r2610, r3732;
}
{
fma.rn.f16x2 r3746, r2754, r3730, r3743;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3750, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3752, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3754, {low, high};
}
{
mul.f16x2 r3755, r3752, r3754;
}
{
mul.f16x2 r3758, r3726, r3750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3726;
mov.b32 r3761, {high, low};
}
{
fma.rn.f16x2 r3763, r3755, r3761, r3758;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3767, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3769, {high, high};
}
{
mul.f16x2 r3771, r3074, r3769;
}
{
neg.f16x2 r3774, r3771;
}
{
fma.rn.f16x2 r3776, r2930, r3767, r3774;
}
{
mul.f16x2 r3780, r2930, r3769;
}
{
fma.rn.f16x2 r3783, r3074, r3767, r3780;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3787, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3789, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3791, {low, high};
}
{
mul.f16x2 r3792, r3789, r3791;
}
{
mul.f16x2 r3795, r3763, r3787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3763;
mov.b32 r3798, {high, low};
}
{
fma.rn.f16x2 r3800, r3792, r3798, r3795;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3804, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3806, {high, high};
}
{
mul.f16x2 r3808, r3394, r3806;
}
{
neg.f16x2 r3811, r3808;
}
{
fma.rn.f16x2 r3813, r3250, r3804, r3811;
}
{
mul.f16x2 r3817, r3250, r3806;
}
{
fma.rn.f16x2 r3820, r3394, r3804, r3817;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3824, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3826, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3828, {low, high};
}
{
mul.f16x2 r3829, r3826, r3828;
}
{
mul.f16x2 r3832, r3800, r3824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3800;
mov.b32 r3835, {high, low};
}
{
fma.rn.f16x2 r3837, r3829, r3835, r3832;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3841, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3843, {high, high};
}
{
mul.f16x2 r3845, r2186, r3843;
}
{
neg.f16x2 r3848, r3845;
}
{
fma.rn.f16x2 r3850, r2042, r3841, r3848;
}
{
mul.f16x2 r3854, r2042, r3843;
}
{
fma.rn.f16x2 r3857, r2186, r3841, r3854;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3861, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3863, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3865, {low, high};
}
{
mul.f16x2 r3866, r3863, r3865;
}
{
mul.f16x2 r3869, r3837, r3861;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3837;
mov.b32 r3872, {high, low};
}
{
fma.rn.f16x2 r3874, r3866, r3872, r3869;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3878, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3880, {high, high};
}
{
mul.f16x2 r3882, r2506, r3880;
}
{
neg.f16x2 r3885, r3882;
}
{
fma.rn.f16x2 r3887, r2362, r3878, r3885;
}
{
mul.f16x2 r3891, r2362, r3880;
}
{
fma.rn.f16x2 r3894, r2506, r3878, r3891;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3898, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3900, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3902, {low, high};
}
{
mul.f16x2 r3903, r3900, r3902;
}
{
mul.f16x2 r3906, r3874, r3898;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3874;
mov.b32 r3909, {high, low};
}
{
fma.rn.f16x2 r3911, r3903, r3909, r3906;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3915, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3917, {high, high};
}
{
mul.f16x2 r3919, r2826, r3917;
}
{
neg.f16x2 r3922, r3919;
}
{
fma.rn.f16x2 r3924, r2682, r3915, r3922;
}
{
mul.f16x2 r3928, r2682, r3917;
}
{
fma.rn.f16x2 r3931, r2826, r3915, r3928;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3935, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3937, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3939, {low, high};
}
{
mul.f16x2 r3940, r3937, r3939;
}
{
mul.f16x2 r3943, r3911, r3935;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3911;
mov.b32 r3946, {high, low};
}
{
fma.rn.f16x2 r3948, r3940, r3946, r3943;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3952, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3954, {high, high};
}
{
mul.f16x2 r3956, r3146, r3954;
}
{
neg.f16x2 r3959, r3956;
}
{
fma.rn.f16x2 r3961, r3002, r3952, r3959;
}
{
mul.f16x2 r3965, r3002, r3954;
}
{
fma.rn.f16x2 r3968, r3146, r3952, r3965;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3972, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r3974, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r3976, {low, high};
}
{
mul.f16x2 r3977, r3974, r3976;
}
{
mul.f16x2 r3980, r3948, r3972;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3948;
mov.b32 r3983, {high, low};
}
{
fma.rn.f16x2 r3985, r3977, r3983, r3980;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r3989, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r3991, {high, high};
}
{
mul.f16x2 r3993, r3466, r3991;
}
{
neg.f16x2 r3996, r3993;
}
{
fma.rn.f16x2 r3998, r3322, r3989, r3996;
}
{
mul.f16x2 r4002, r3322, r3991;
}
{
fma.rn.f16x2 r4005, r3466, r3989, r4002;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4009, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4011, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4013, {low, high};
}
{
mul.f16x2 r4014, r4011, r4013;
}
{
mul.f16x2 r4017, r3985, r4009;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3985;
mov.b32 r4020, {high, low};
}
{
fma.rn.f16x2 r4022, r4014, r4020, r4017;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4026, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4028, {high, high};
}
{
mul.f16x2 r4030, r2222, r4028;
}
{
neg.f16x2 r4033, r4030;
}
{
fma.rn.f16x2 r4035, r2078, r4026, r4033;
}
{
mul.f16x2 r4039, r2078, r4028;
}
{
fma.rn.f16x2 r4042, r2222, r4026, r4039;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4046, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4048, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4050, {low, high};
}
{
mul.f16x2 r4051, r4048, r4050;
}
{
mul.f16x2 r4054, r4022, r4046;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4022;
mov.b32 r4057, {high, low};
}
{
fma.rn.f16x2 r4059, r4051, r4057, r4054;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4063, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4065, {high, high};
}
{
mul.f16x2 r4067, r2542, r4065;
}
{
neg.f16x2 r4070, r4067;
}
{
fma.rn.f16x2 r4072, r2398, r4063, r4070;
}
{
mul.f16x2 r4076, r2398, r4065;
}
{
fma.rn.f16x2 r4079, r2542, r4063, r4076;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4083, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4085, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4087, {low, high};
}
{
mul.f16x2 r4088, r4085, r4087;
}
{
mul.f16x2 r4091, r4059, r4083;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4059;
mov.b32 r4094, {high, low};
}
{
fma.rn.f16x2 r4096, r4088, r4094, r4091;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4100, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4102, {high, high};
}
{
mul.f16x2 r4104, r2862, r4102;
}
{
neg.f16x2 r4107, r4104;
}
{
fma.rn.f16x2 r4109, r2718, r4100, r4107;
}
{
mul.f16x2 r4113, r2718, r4102;
}
{
fma.rn.f16x2 r4116, r2862, r4100, r4113;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4122, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4124, {low, high};
}
{
mul.f16x2 r4125, r4122, r4124;
}
{
mul.f16x2 r4128, r4096, r4120;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4096;
mov.b32 r4131, {high, low};
}
{
fma.rn.f16x2 r4133, r4125, r4131, r4128;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4137, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4139, {high, high};
}
{
mul.f16x2 r4141, r3182, r4139;
}
{
neg.f16x2 r4144, r4141;
}
{
fma.rn.f16x2 r4146, r3038, r4137, r4144;
}
{
mul.f16x2 r4150, r3038, r4139;
}
{
fma.rn.f16x2 r4153, r3182, r4137, r4150;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4157, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4159, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4161, {low, high};
}
{
mul.f16x2 r4162, r4159, r4161;
}
{
mul.f16x2 r4165, r4133, r4157;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4133;
mov.b32 r4168, {high, low};
}
{
fma.rn.f16x2 r4170, r4162, r4168, r4165;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4174, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4176, {high, high};
}
{
mul.f16x2 r4178, r3502, r4176;
}
{
neg.f16x2 r4181, r4178;
}
{
fma.rn.f16x2 r4183, r3358, r4174, r4181;
}
{
mul.f16x2 r4187, r3358, r4176;
}
{
fma.rn.f16x2 r4190, r3502, r4174, r4187;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4194, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4196, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4198, {low, high};
}
{
mul.f16x2 r4199, r4196, r4198;
}
{
mul.f16x2 r4202, r4170, r4194;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4170;
mov.b32 r4205, {high, low};
}
{
fma.rn.f16x2 r4207, r4199, r4205, r4202;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4211, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4213, {high, high};
}
{
mul.f16x2 r4215, r2150, r4213;
}
{
neg.f16x2 r4218, r4215;
}
{
fma.rn.f16x2 r4220, r2006, r4211, r4218;
}
{
mul.f16x2 r4224, r2006, r4213;
}
{
fma.rn.f16x2 r4227, r2150, r4211, r4224;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4231, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4233, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4235, {low, high};
}
{
mul.f16x2 r4236, r4233, r4235;
}
{
mul.f16x2 r4239, r4207, r4231;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4207;
mov.b32 r4242, {high, low};
}
{
fma.rn.f16x2 r4244, r4236, r4242, r4239;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4248, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4250, {high, high};
}
{
mul.f16x2 r4252, r2470, r4250;
}
{
neg.f16x2 r4255, r4252;
}
{
fma.rn.f16x2 r4257, r2326, r4248, r4255;
}
{
mul.f16x2 r4261, r2326, r4250;
}
{
fma.rn.f16x2 r4264, r2470, r4248, r4261;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4268, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4270, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4272, {low, high};
}
{
mul.f16x2 r4273, r4270, r4272;
}
{
mul.f16x2 r4276, r4244, r4268;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4244;
mov.b32 r4279, {high, low};
}
{
fma.rn.f16x2 r4281, r4273, r4279, r4276;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4285, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4287, {high, high};
}
{
mul.f16x2 r4289, r2790, r4287;
}
{
neg.f16x2 r4292, r4289;
}
{
fma.rn.f16x2 r4294, r2646, r4285, r4292;
}
{
mul.f16x2 r4298, r2646, r4287;
}
{
fma.rn.f16x2 r4301, r2790, r4285, r4298;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4305, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4307, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4309, {low, high};
}
{
mul.f16x2 r4310, r4307, r4309;
}
{
mul.f16x2 r4313, r4281, r4305;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4281;
mov.b32 r4316, {high, low};
}
{
fma.rn.f16x2 r4318, r4310, r4316, r4313;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4322, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4324, {high, high};
}
{
mul.f16x2 r4326, r3110, r4324;
}
{
neg.f16x2 r4329, r4326;
}
{
fma.rn.f16x2 r4331, r2966, r4322, r4329;
}
{
mul.f16x2 r4335, r2966, r4324;
}
{
fma.rn.f16x2 r4338, r3110, r4322, r4335;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4342, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r3505;
mov.b32 r4344, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r4346, {low, high};
}
{
mul.f16x2 r4347, r4344, r4346;
}
{
mul.f16x2 r4350, r4318, r4342;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4318;
mov.b32 r4353, {high, low};
}
{
fma.rn.f16x2 r4355, r4347, r4353, r4350;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4355;
mov.b32 r4359, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r4355;
mov.b32 r4361, {high, high};
}
{
mul.f16x2 r4363, r3430, r4361;
}
{
neg.f16x2 r4366, r4363;
}
{
fma.rn.f16x2 r4368, r3286, r4359, r4366;
}
{
mul.f16x2 r4372, r3286, r4361;
}
{
fma.rn.f16x2 r4375, r3430, r4359, r4372;
}
barrier.sync 0;
mad.lo.s32 r10399, r10397, 100, r10398;
st.shared.u32 [r10399], r1922;
st.shared.u32 [r10399+4], r3517;
st.shared.u32 [r10399+8], r3554;
st.shared.u32 [r10399+12], r3591;
st.shared.u32 [r10399+16], r3628;
st.shared.u32 [r10399+20], r3665;
st.shared.u32 [r10399+24], r3702;
st.shared.u32 [r10399+28], r3739;
st.shared.u32 [r10399+32], r3776;
st.shared.u32 [r10399+36], r3813;
st.shared.u32 [r10399+40], r3850;
st.shared.u32 [r10399+44], r3887;
st.shared.u32 [r10399+48], r3924;
st.shared.u32 [r10399+52], r3961;
st.shared.u32 [r10399+56], r3998;
st.shared.u32 [r10399+60], r4035;
st.shared.u32 [r10399+64], r4072;
st.shared.u32 [r10399+68], r4109;
st.shared.u32 [r10399+72], r4146;
st.shared.u32 [r10399+76], r4183;
st.shared.u32 [r10399+80], r4220;
st.shared.u32 [r10399+84], r4257;
st.shared.u32 [r10399+88], r4294;
st.shared.u32 [r10399+92], r4331;
st.shared.u32 [r10399+96], r4368;
barrier.sync 0;
mad.lo.s32 r10400, r10397, -96, r10399;
ld.shared.u32 r4408, [r10400];
ld.shared.u32 r4728, [r10400+500];
ld.shared.u32 r5048, [r10400+1000];
ld.shared.u32 r5368, [r10400+1500];
ld.shared.u32 r5688, [r10400+2000];
ld.shared.u32 r4405, [r10400+2500];
ld.shared.u32 r4725, [r10400+3000];
ld.shared.u32 r5045, [r10400+3500];
ld.shared.u32 r5365, [r10400+4000];
ld.shared.u32 r5685, [r10400+4500];
ld.shared.u32 r4411, [r10400+5000];
ld.shared.u32 r4731, [r10400+5500];
ld.shared.u32 r5051, [r10400+6000];
ld.shared.u32 r5371, [r10400+6500];
ld.shared.u32 r5691, [r10400+7000];
ld.shared.u32 r4412, [r10400+7500];
ld.shared.u32 r4732, [r10400+8000];
ld.shared.u32 r5052, [r10400+8500];
ld.shared.u32 r5372, [r10400+9000];
ld.shared.u32 r5692, [r10400+9500];
ld.shared.u32 r4406, [r10400+10000];
ld.shared.u32 r4726, [r10400+10500];
ld.shared.u32 r5046, [r10400+11000];
ld.shared.u32 r5366, [r10400+11500];
ld.shared.u32 r5686, [r10400+12000];
barrier.sync 0;
st.shared.u32 [r10399], r1934;
st.shared.u32 [r10399+4], r3524;
st.shared.u32 [r10399+8], r3561;
st.shared.u32 [r10399+12], r3598;
st.shared.u32 [r10399+16], r3635;
st.shared.u32 [r10399+20], r3672;
st.shared.u32 [r10399+24], r3709;
st.shared.u32 [r10399+28], r3746;
st.shared.u32 [r10399+32], r3783;
st.shared.u32 [r10399+36], r3820;
st.shared.u32 [r10399+40], r3857;
st.shared.u32 [r10399+44], r3894;
st.shared.u32 [r10399+48], r3931;
st.shared.u32 [r10399+52], r3968;
st.shared.u32 [r10399+56], r4005;
st.shared.u32 [r10399+60], r4042;
st.shared.u32 [r10399+64], r4079;
st.shared.u32 [r10399+68], r4116;
st.shared.u32 [r10399+72], r4153;
st.shared.u32 [r10399+76], r4190;
st.shared.u32 [r10399+80], r4227;
st.shared.u32 [r10399+84], r4264;
st.shared.u32 [r10399+88], r4301;
st.shared.u32 [r10399+92], r4338;
st.shared.u32 [r10399+96], r4375;
barrier.sync 0;
ld.shared.u32 r4420, [r10400];
ld.shared.u32 r4740, [r10400+500];
ld.shared.u32 r5060, [r10400+1000];
ld.shared.u32 r5380, [r10400+1500];
ld.shared.u32 r5700, [r10400+2000];
ld.shared.u32 r4417, [r10400+2500];
ld.shared.u32 r4737, [r10400+3000];
ld.shared.u32 r5057, [r10400+3500];
ld.shared.u32 r5377, [r10400+4000];
ld.shared.u32 r5697, [r10400+4500];
ld.shared.u32 r4423, [r10400+5000];
ld.shared.u32 r4743, [r10400+5500];
ld.shared.u32 r5063, [r10400+6000];
ld.shared.u32 r5383, [r10400+6500];
ld.shared.u32 r5703, [r10400+7000];
ld.shared.u32 r4424, [r10400+7500];
ld.shared.u32 r4744, [r10400+8000];
ld.shared.u32 r5064, [r10400+8500];
ld.shared.u32 r5384, [r10400+9000];
ld.shared.u32 r5704, [r10400+9500];
ld.shared.u32 r4418, [r10400+10000];
ld.shared.u32 r4738, [r10400+10500];
ld.shared.u32 r5058, [r10400+11000];
ld.shared.u32 r5378, [r10400+11500];
ld.shared.u32 r5698, [r10400+12000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4401, {low, high};
}
{
neg.f16x2 r4402, r4401;
}
{
add.f16x2 r4404, r4405, r4406;
}
{
add.f16x2 r4407, r4408, r4404;
}
{
add.f16x2 r4410, r4411, r4412;
}
{
add.f16x2 r4413, r4407, r4410;
}
{
add.f16x2 r4416, r4417, r4418;
}
{
add.f16x2 r4419, r4420, r4416;
}
{
add.f16x2 r4422, r4423, r4424;
}
{
add.f16x2 r4425, r4419, r4422;
}
{
add.f16x2 r4428, r4405, r4406;
}
{
mul.f16x2 r4431, r4428, r4396;
}
{
add.f16x2 r4434, r4408, r4431;
}
{
add.f16x2 r4437, r4411, r4412;
}
{
mul.f16x2 r4440, r4437, r4398;
}
{
add.f16x2 r4443, r4434, r4440;
}
{
sub.f16x2 r4446, r4417, r4418;
}
{
mul.f16x2 r4449, r4446, r4397;
}
{
sub.f16x2 r4452, r4423, r4424;
}
{
mul.f16x2 r4455, r4452, r4399;
}
{
add.f16x2 r4458, r4449, r4455;
}
{
sub.f16x2 r4461, r4443, r4458;
}
{
add.f16x2 r4464, r4405, r4406;
}
{
mul.f16x2 r4467, r4464, r4396;
}
{
add.f16x2 r4470, r4408, r4467;
}
{
add.f16x2 r4473, r4411, r4412;
}
{
mul.f16x2 r4476, r4473, r4398;
}
{
add.f16x2 r4479, r4470, r4476;
}
{
sub.f16x2 r4482, r4417, r4418;
}
{
mul.f16x2 r4485, r4482, r4397;
}
{
sub.f16x2 r4488, r4423, r4424;
}
{
mul.f16x2 r4491, r4488, r4399;
}
{
add.f16x2 r4494, r4485, r4491;
}
{
add.f16x2 r4497, r4479, r4494;
}
{
add.f16x2 r4500, r4405, r4406;
}
{
mul.f16x2 r4503, r4500, r4398;
}
{
add.f16x2 r4506, r4408, r4503;
}
{
add.f16x2 r4509, r4411, r4412;
}
{
mul.f16x2 r4512, r4509, r4400;
}
{
add.f16x2 r4515, r4506, r4512;
}
{
sub.f16x2 r4518, r4417, r4418;
}
{
mul.f16x2 r4521, r4518, r4399;
}
{
sub.f16x2 r4524, r4423, r4424;
}
{
mul.f16x2 r4527, r4524, r4402;
}
{
add.f16x2 r4530, r4521, r4527;
}
{
sub.f16x2 r4533, r4515, r4530;
}
{
add.f16x2 r4536, r4405, r4406;
}
{
mul.f16x2 r4539, r4536, r4398;
}
{
add.f16x2 r4542, r4408, r4539;
}
{
add.f16x2 r4545, r4411, r4412;
}
{
mul.f16x2 r4548, r4545, r4400;
}
{
add.f16x2 r4551, r4542, r4548;
}
{
sub.f16x2 r4554, r4417, r4418;
}
{
mul.f16x2 r4557, r4554, r4399;
}
{
sub.f16x2 r4560, r4423, r4424;
}
{
mul.f16x2 r4563, r4560, r4402;
}
{
add.f16x2 r4566, r4557, r4563;
}
{
add.f16x2 r4569, r4551, r4566;
}
{
add.f16x2 r4572, r4417, r4418;
}
{
mul.f16x2 r4575, r4572, r4396;
}
{
add.f16x2 r4578, r4420, r4575;
}
{
add.f16x2 r4581, r4423, r4424;
}
{
mul.f16x2 r4584, r4581, r4398;
}
{
add.f16x2 r4587, r4578, r4584;
}
{
sub.f16x2 r4590, r4405, r4406;
}
{
mul.f16x2 r4593, r4590, r4397;
}
{
sub.f16x2 r4596, r4411, r4412;
}
{
mul.f16x2 r4599, r4596, r4399;
}
{
add.f16x2 r4602, r4593, r4599;
}
{
add.f16x2 r4605, r4587, r4602;
}
{
add.f16x2 r4608, r4417, r4418;
}
{
mul.f16x2 r4611, r4608, r4396;
}
{
add.f16x2 r4614, r4420, r4611;
}
{
add.f16x2 r4617, r4423, r4424;
}
{
mul.f16x2 r4620, r4617, r4398;
}
{
add.f16x2 r4623, r4614, r4620;
}
{
sub.f16x2 r4626, r4405, r4406;
}
{
mul.f16x2 r4629, r4626, r4397;
}
{
sub.f16x2 r4632, r4411, r4412;
}
{
mul.f16x2 r4635, r4632, r4399;
}
{
add.f16x2 r4638, r4629, r4635;
}
{
sub.f16x2 r4641, r4623, r4638;
}
{
add.f16x2 r4644, r4417, r4418;
}
{
mul.f16x2 r4647, r4644, r4398;
}
{
add.f16x2 r4650, r4420, r4647;
}
{
add.f16x2 r4653, r4423, r4424;
}
{
mul.f16x2 r4656, r4653, r4400;
}
{
add.f16x2 r4659, r4650, r4656;
}
{
sub.f16x2 r4662, r4405, r4406;
}
{
mul.f16x2 r4665, r4662, r4399;
}
{
sub.f16x2 r4668, r4411, r4412;
}
{
mul.f16x2 r4671, r4668, r4402;
}
{
add.f16x2 r4674, r4665, r4671;
}
{
add.f16x2 r4677, r4659, r4674;
}
{
add.f16x2 r4680, r4417, r4418;
}
{
mul.f16x2 r4683, r4680, r4398;
}
{
add.f16x2 r4686, r4420, r4683;
}
{
add.f16x2 r4689, r4423, r4424;
}
{
mul.f16x2 r4692, r4689, r4400;
}
{
add.f16x2 r4695, r4686, r4692;
}
{
sub.f16x2 r4698, r4405, r4406;
}
{
mul.f16x2 r4701, r4698, r4399;
}
{
sub.f16x2 r4704, r4411, r4412;
}
{
mul.f16x2 r4707, r4704, r4402;
}
{
add.f16x2 r4710, r4701, r4707;
}
{
sub.f16x2 r4713, r4695, r4710;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r4718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r4719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r4720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r4721, {low, high};
}
{
neg.f16x2 r4722, r4721;
}
{
add.f16x2 r4724, r4725, r4726;
}
{
add.f16x2 r4727, r4728, r4724;
}
{
add.f16x2 r4730, r4731, r4732;
}
{
add.f16x2 r4733, r4727, r4730;
}
{
add.f16x2 r4736, r4737, r4738;
}
{
add.f16x2 r4739, r4740, r4736;
}
{
add.f16x2 r4742, r4743, r4744;
}
{
add.f16x2 r4745, r4739, r4742;
}
{
add.f16x2 r4748, r4725, r4726;
}
{
mul.f16x2 r4751, r4748, r4716;
}
{
add.f16x2 r4754, r4728, r4751;
}
{
add.f16x2 r4757, r4731, r4732;
}
{
mul.f16x2 r4760, r4757, r4718;
}
{
add.f16x2 r4763, r4754, r4760;
}
{
sub.f16x2 r4766, r4737, r4738;
}
{
mul.f16x2 r4769, r4766, r4717;
}
{
sub.f16x2 r4772, r4743, r4744;
}
{
mul.f16x2 r4775, r4772, r4719;
}
{
add.f16x2 r4778, r4769, r4775;
}
{
sub.f16x2 r4781, r4763, r4778;
}
{
add.f16x2 r4784, r4725, r4726;
}
{
mul.f16x2 r4787, r4784, r4716;
}
{
add.f16x2 r4790, r4728, r4787;
}
{
add.f16x2 r4793, r4731, r4732;
}
{
mul.f16x2 r4796, r4793, r4718;
}
{
add.f16x2 r4799, r4790, r4796;
}
{
sub.f16x2 r4802, r4737, r4738;
}
{
mul.f16x2 r4805, r4802, r4717;
}
{
sub.f16x2 r4808, r4743, r4744;
}
{
mul.f16x2 r4811, r4808, r4719;
}
{
add.f16x2 r4814, r4805, r4811;
}
{
add.f16x2 r4817, r4799, r4814;
}
{
add.f16x2 r4820, r4725, r4726;
}
{
mul.f16x2 r4823, r4820, r4718;
}
{
add.f16x2 r4826, r4728, r4823;
}
{
add.f16x2 r4829, r4731, r4732;
}
{
mul.f16x2 r4832, r4829, r4720;
}
{
add.f16x2 r4835, r4826, r4832;
}
{
sub.f16x2 r4838, r4737, r4738;
}
{
mul.f16x2 r4841, r4838, r4719;
}
{
sub.f16x2 r4844, r4743, r4744;
}
{
mul.f16x2 r4847, r4844, r4722;
}
{
add.f16x2 r4850, r4841, r4847;
}
{
sub.f16x2 r4853, r4835, r4850;
}
{
add.f16x2 r4856, r4725, r4726;
}
{
mul.f16x2 r4859, r4856, r4718;
}
{
add.f16x2 r4862, r4728, r4859;
}
{
add.f16x2 r4865, r4731, r4732;
}
{
mul.f16x2 r4868, r4865, r4720;
}
{
add.f16x2 r4871, r4862, r4868;
}
{
sub.f16x2 r4874, r4737, r4738;
}
{
mul.f16x2 r4877, r4874, r4719;
}
{
sub.f16x2 r4880, r4743, r4744;
}
{
mul.f16x2 r4883, r4880, r4722;
}
{
add.f16x2 r4886, r4877, r4883;
}
{
add.f16x2 r4889, r4871, r4886;
}
{
add.f16x2 r4892, r4737, r4738;
}
{
mul.f16x2 r4895, r4892, r4716;
}
{
add.f16x2 r4898, r4740, r4895;
}
{
add.f16x2 r4901, r4743, r4744;
}
{
mul.f16x2 r4904, r4901, r4718;
}
{
add.f16x2 r4907, r4898, r4904;
}
{
sub.f16x2 r4910, r4725, r4726;
}
{
mul.f16x2 r4913, r4910, r4717;
}
{
sub.f16x2 r4916, r4731, r4732;
}
{
mul.f16x2 r4919, r4916, r4719;
}
{
add.f16x2 r4922, r4913, r4919;
}
{
add.f16x2 r4925, r4907, r4922;
}
{
add.f16x2 r4928, r4737, r4738;
}
{
mul.f16x2 r4931, r4928, r4716;
}
{
add.f16x2 r4934, r4740, r4931;
}
{
add.f16x2 r4937, r4743, r4744;
}
{
mul.f16x2 r4940, r4937, r4718;
}
{
add.f16x2 r4943, r4934, r4940;
}
{
sub.f16x2 r4946, r4725, r4726;
}
{
mul.f16x2 r4949, r4946, r4717;
}
{
sub.f16x2 r4952, r4731, r4732;
}
{
mul.f16x2 r4955, r4952, r4719;
}
{
add.f16x2 r4958, r4949, r4955;
}
{
sub.f16x2 r4961, r4943, r4958;
}
{
add.f16x2 r4964, r4737, r4738;
}
{
mul.f16x2 r4967, r4964, r4718;
}
{
add.f16x2 r4970, r4740, r4967;
}
{
add.f16x2 r4973, r4743, r4744;
}
{
mul.f16x2 r4976, r4973, r4720;
}
{
add.f16x2 r4979, r4970, r4976;
}
{
sub.f16x2 r4982, r4725, r4726;
}
{
mul.f16x2 r4985, r4982, r4719;
}
{
sub.f16x2 r4988, r4731, r4732;
}
{
mul.f16x2 r4991, r4988, r4722;
}
{
add.f16x2 r4994, r4985, r4991;
}
{
add.f16x2 r4997, r4979, r4994;
}
{
add.f16x2 r5000, r4737, r4738;
}
{
mul.f16x2 r5003, r5000, r4718;
}
{
add.f16x2 r5006, r4740, r5003;
}
{
add.f16x2 r5009, r4743, r4744;
}
{
mul.f16x2 r5012, r5009, r4720;
}
{
add.f16x2 r5015, r5006, r5012;
}
{
sub.f16x2 r5018, r4725, r4726;
}
{
mul.f16x2 r5021, r5018, r4719;
}
{
sub.f16x2 r5024, r4731, r4732;
}
{
mul.f16x2 r5027, r5024, r4722;
}
{
add.f16x2 r5030, r5021, r5027;
}
{
sub.f16x2 r5033, r5015, r5030;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5038, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5039, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5041, {low, high};
}
{
neg.f16x2 r5042, r5041;
}
{
add.f16x2 r5044, r5045, r5046;
}
{
add.f16x2 r5047, r5048, r5044;
}
{
add.f16x2 r5050, r5051, r5052;
}
{
add.f16x2 r5053, r5047, r5050;
}
{
add.f16x2 r5056, r5057, r5058;
}
{
add.f16x2 r5059, r5060, r5056;
}
{
add.f16x2 r5062, r5063, r5064;
}
{
add.f16x2 r5065, r5059, r5062;
}
{
add.f16x2 r5068, r5045, r5046;
}
{
mul.f16x2 r5071, r5068, r5036;
}
{
add.f16x2 r5074, r5048, r5071;
}
{
add.f16x2 r5077, r5051, r5052;
}
{
mul.f16x2 r5080, r5077, r5038;
}
{
add.f16x2 r5083, r5074, r5080;
}
{
sub.f16x2 r5086, r5057, r5058;
}
{
mul.f16x2 r5089, r5086, r5037;
}
{
sub.f16x2 r5092, r5063, r5064;
}
{
mul.f16x2 r5095, r5092, r5039;
}
{
add.f16x2 r5098, r5089, r5095;
}
{
sub.f16x2 r5101, r5083, r5098;
}
{
add.f16x2 r5104, r5045, r5046;
}
{
mul.f16x2 r5107, r5104, r5036;
}
{
add.f16x2 r5110, r5048, r5107;
}
{
add.f16x2 r5113, r5051, r5052;
}
{
mul.f16x2 r5116, r5113, r5038;
}
{
add.f16x2 r5119, r5110, r5116;
}
{
sub.f16x2 r5122, r5057, r5058;
}
{
mul.f16x2 r5125, r5122, r5037;
}
{
sub.f16x2 r5128, r5063, r5064;
}
{
mul.f16x2 r5131, r5128, r5039;
}
{
add.f16x2 r5134, r5125, r5131;
}
{
add.f16x2 r5137, r5119, r5134;
}
{
add.f16x2 r5140, r5045, r5046;
}
{
mul.f16x2 r5143, r5140, r5038;
}
{
add.f16x2 r5146, r5048, r5143;
}
{
add.f16x2 r5149, r5051, r5052;
}
{
mul.f16x2 r5152, r5149, r5040;
}
{
add.f16x2 r5155, r5146, r5152;
}
{
sub.f16x2 r5158, r5057, r5058;
}
{
mul.f16x2 r5161, r5158, r5039;
}
{
sub.f16x2 r5164, r5063, r5064;
}
{
mul.f16x2 r5167, r5164, r5042;
}
{
add.f16x2 r5170, r5161, r5167;
}
{
sub.f16x2 r5173, r5155, r5170;
}
{
add.f16x2 r5176, r5045, r5046;
}
{
mul.f16x2 r5179, r5176, r5038;
}
{
add.f16x2 r5182, r5048, r5179;
}
{
add.f16x2 r5185, r5051, r5052;
}
{
mul.f16x2 r5188, r5185, r5040;
}
{
add.f16x2 r5191, r5182, r5188;
}
{
sub.f16x2 r5194, r5057, r5058;
}
{
mul.f16x2 r5197, r5194, r5039;
}
{
sub.f16x2 r5200, r5063, r5064;
}
{
mul.f16x2 r5203, r5200, r5042;
}
{
add.f16x2 r5206, r5197, r5203;
}
{
add.f16x2 r5209, r5191, r5206;
}
{
add.f16x2 r5212, r5057, r5058;
}
{
mul.f16x2 r5215, r5212, r5036;
}
{
add.f16x2 r5218, r5060, r5215;
}
{
add.f16x2 r5221, r5063, r5064;
}
{
mul.f16x2 r5224, r5221, r5038;
}
{
add.f16x2 r5227, r5218, r5224;
}
{
sub.f16x2 r5230, r5045, r5046;
}
{
mul.f16x2 r5233, r5230, r5037;
}
{
sub.f16x2 r5236, r5051, r5052;
}
{
mul.f16x2 r5239, r5236, r5039;
}
{
add.f16x2 r5242, r5233, r5239;
}
{
add.f16x2 r5245, r5227, r5242;
}
{
add.f16x2 r5248, r5057, r5058;
}
{
mul.f16x2 r5251, r5248, r5036;
}
{
add.f16x2 r5254, r5060, r5251;
}
{
add.f16x2 r5257, r5063, r5064;
}
{
mul.f16x2 r5260, r5257, r5038;
}
{
add.f16x2 r5263, r5254, r5260;
}
{
sub.f16x2 r5266, r5045, r5046;
}
{
mul.f16x2 r5269, r5266, r5037;
}
{
sub.f16x2 r5272, r5051, r5052;
}
{
mul.f16x2 r5275, r5272, r5039;
}
{
add.f16x2 r5278, r5269, r5275;
}
{
sub.f16x2 r5281, r5263, r5278;
}
{
add.f16x2 r5284, r5057, r5058;
}
{
mul.f16x2 r5287, r5284, r5038;
}
{
add.f16x2 r5290, r5060, r5287;
}
{
add.f16x2 r5293, r5063, r5064;
}
{
mul.f16x2 r5296, r5293, r5040;
}
{
add.f16x2 r5299, r5290, r5296;
}
{
sub.f16x2 r5302, r5045, r5046;
}
{
mul.f16x2 r5305, r5302, r5039;
}
{
sub.f16x2 r5308, r5051, r5052;
}
{
mul.f16x2 r5311, r5308, r5042;
}
{
add.f16x2 r5314, r5305, r5311;
}
{
add.f16x2 r5317, r5299, r5314;
}
{
add.f16x2 r5320, r5057, r5058;
}
{
mul.f16x2 r5323, r5320, r5038;
}
{
add.f16x2 r5326, r5060, r5323;
}
{
add.f16x2 r5329, r5063, r5064;
}
{
mul.f16x2 r5332, r5329, r5040;
}
{
add.f16x2 r5335, r5326, r5332;
}
{
sub.f16x2 r5338, r5045, r5046;
}
{
mul.f16x2 r5341, r5338, r5039;
}
{
sub.f16x2 r5344, r5051, r5052;
}
{
mul.f16x2 r5347, r5344, r5042;
}
{
add.f16x2 r5350, r5341, r5347;
}
{
sub.f16x2 r5353, r5335, r5350;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5361, {low, high};
}
{
neg.f16x2 r5362, r5361;
}
{
add.f16x2 r5364, r5365, r5366;
}
{
add.f16x2 r5367, r5368, r5364;
}
{
add.f16x2 r5370, r5371, r5372;
}
{
add.f16x2 r5373, r5367, r5370;
}
{
add.f16x2 r5376, r5377, r5378;
}
{
add.f16x2 r5379, r5380, r5376;
}
{
add.f16x2 r5382, r5383, r5384;
}
{
add.f16x2 r5385, r5379, r5382;
}
{
add.f16x2 r5388, r5365, r5366;
}
{
mul.f16x2 r5391, r5388, r5356;
}
{
add.f16x2 r5394, r5368, r5391;
}
{
add.f16x2 r5397, r5371, r5372;
}
{
mul.f16x2 r5400, r5397, r5358;
}
{
add.f16x2 r5403, r5394, r5400;
}
{
sub.f16x2 r5406, r5377, r5378;
}
{
mul.f16x2 r5409, r5406, r5357;
}
{
sub.f16x2 r5412, r5383, r5384;
}
{
mul.f16x2 r5415, r5412, r5359;
}
{
add.f16x2 r5418, r5409, r5415;
}
{
sub.f16x2 r5421, r5403, r5418;
}
{
add.f16x2 r5424, r5365, r5366;
}
{
mul.f16x2 r5427, r5424, r5356;
}
{
add.f16x2 r5430, r5368, r5427;
}
{
add.f16x2 r5433, r5371, r5372;
}
{
mul.f16x2 r5436, r5433, r5358;
}
{
add.f16x2 r5439, r5430, r5436;
}
{
sub.f16x2 r5442, r5377, r5378;
}
{
mul.f16x2 r5445, r5442, r5357;
}
{
sub.f16x2 r5448, r5383, r5384;
}
{
mul.f16x2 r5451, r5448, r5359;
}
{
add.f16x2 r5454, r5445, r5451;
}
{
add.f16x2 r5457, r5439, r5454;
}
{
add.f16x2 r5460, r5365, r5366;
}
{
mul.f16x2 r5463, r5460, r5358;
}
{
add.f16x2 r5466, r5368, r5463;
}
{
add.f16x2 r5469, r5371, r5372;
}
{
mul.f16x2 r5472, r5469, r5360;
}
{
add.f16x2 r5475, r5466, r5472;
}
{
sub.f16x2 r5478, r5377, r5378;
}
{
mul.f16x2 r5481, r5478, r5359;
}
{
sub.f16x2 r5484, r5383, r5384;
}
{
mul.f16x2 r5487, r5484, r5362;
}
{
add.f16x2 r5490, r5481, r5487;
}
{
sub.f16x2 r5493, r5475, r5490;
}
{
add.f16x2 r5496, r5365, r5366;
}
{
mul.f16x2 r5499, r5496, r5358;
}
{
add.f16x2 r5502, r5368, r5499;
}
{
add.f16x2 r5505, r5371, r5372;
}
{
mul.f16x2 r5508, r5505, r5360;
}
{
add.f16x2 r5511, r5502, r5508;
}
{
sub.f16x2 r5514, r5377, r5378;
}
{
mul.f16x2 r5517, r5514, r5359;
}
{
sub.f16x2 r5520, r5383, r5384;
}
{
mul.f16x2 r5523, r5520, r5362;
}
{
add.f16x2 r5526, r5517, r5523;
}
{
add.f16x2 r5529, r5511, r5526;
}
{
add.f16x2 r5532, r5377, r5378;
}
{
mul.f16x2 r5535, r5532, r5356;
}
{
add.f16x2 r5538, r5380, r5535;
}
{
add.f16x2 r5541, r5383, r5384;
}
{
mul.f16x2 r5544, r5541, r5358;
}
{
add.f16x2 r5547, r5538, r5544;
}
{
sub.f16x2 r5550, r5365, r5366;
}
{
mul.f16x2 r5553, r5550, r5357;
}
{
sub.f16x2 r5556, r5371, r5372;
}
{
mul.f16x2 r5559, r5556, r5359;
}
{
add.f16x2 r5562, r5553, r5559;
}
{
add.f16x2 r5565, r5547, r5562;
}
{
add.f16x2 r5568, r5377, r5378;
}
{
mul.f16x2 r5571, r5568, r5356;
}
{
add.f16x2 r5574, r5380, r5571;
}
{
add.f16x2 r5577, r5383, r5384;
}
{
mul.f16x2 r5580, r5577, r5358;
}
{
add.f16x2 r5583, r5574, r5580;
}
{
sub.f16x2 r5586, r5365, r5366;
}
{
mul.f16x2 r5589, r5586, r5357;
}
{
sub.f16x2 r5592, r5371, r5372;
}
{
mul.f16x2 r5595, r5592, r5359;
}
{
add.f16x2 r5598, r5589, r5595;
}
{
sub.f16x2 r5601, r5583, r5598;
}
{
add.f16x2 r5604, r5377, r5378;
}
{
mul.f16x2 r5607, r5604, r5358;
}
{
add.f16x2 r5610, r5380, r5607;
}
{
add.f16x2 r5613, r5383, r5384;
}
{
mul.f16x2 r5616, r5613, r5360;
}
{
add.f16x2 r5619, r5610, r5616;
}
{
sub.f16x2 r5622, r5365, r5366;
}
{
mul.f16x2 r5625, r5622, r5359;
}
{
sub.f16x2 r5628, r5371, r5372;
}
{
mul.f16x2 r5631, r5628, r5362;
}
{
add.f16x2 r5634, r5625, r5631;
}
{
add.f16x2 r5637, r5619, r5634;
}
{
add.f16x2 r5640, r5377, r5378;
}
{
mul.f16x2 r5643, r5640, r5358;
}
{
add.f16x2 r5646, r5380, r5643;
}
{
add.f16x2 r5649, r5383, r5384;
}
{
mul.f16x2 r5652, r5649, r5360;
}
{
add.f16x2 r5655, r5646, r5652;
}
{
sub.f16x2 r5658, r5365, r5366;
}
{
mul.f16x2 r5661, r5658, r5359;
}
{
sub.f16x2 r5664, r5371, r5372;
}
{
mul.f16x2 r5667, r5664, r5362;
}
{
add.f16x2 r5670, r5661, r5667;
}
{
sub.f16x2 r5673, r5655, r5670;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r5678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r5679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r5680, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r5681, {low, high};
}
{
neg.f16x2 r5682, r5681;
}
{
add.f16x2 r5684, r5685, r5686;
}
{
add.f16x2 r5687, r5688, r5684;
}
{
add.f16x2 r5690, r5691, r5692;
}
{
add.f16x2 r5693, r5687, r5690;
}
{
add.f16x2 r5696, r5697, r5698;
}
{
add.f16x2 r5699, r5700, r5696;
}
{
add.f16x2 r5702, r5703, r5704;
}
{
add.f16x2 r5705, r5699, r5702;
}
{
add.f16x2 r5708, r5685, r5686;
}
{
mul.f16x2 r5711, r5708, r5676;
}
{
add.f16x2 r5714, r5688, r5711;
}
{
add.f16x2 r5717, r5691, r5692;
}
{
mul.f16x2 r5720, r5717, r5678;
}
{
add.f16x2 r5723, r5714, r5720;
}
{
sub.f16x2 r5726, r5697, r5698;
}
{
mul.f16x2 r5729, r5726, r5677;
}
{
sub.f16x2 r5732, r5703, r5704;
}
{
mul.f16x2 r5735, r5732, r5679;
}
{
add.f16x2 r5738, r5729, r5735;
}
{
sub.f16x2 r5741, r5723, r5738;
}
{
add.f16x2 r5744, r5685, r5686;
}
{
mul.f16x2 r5747, r5744, r5676;
}
{
add.f16x2 r5750, r5688, r5747;
}
{
add.f16x2 r5753, r5691, r5692;
}
{
mul.f16x2 r5756, r5753, r5678;
}
{
add.f16x2 r5759, r5750, r5756;
}
{
sub.f16x2 r5762, r5697, r5698;
}
{
mul.f16x2 r5765, r5762, r5677;
}
{
sub.f16x2 r5768, r5703, r5704;
}
{
mul.f16x2 r5771, r5768, r5679;
}
{
add.f16x2 r5774, r5765, r5771;
}
{
add.f16x2 r5777, r5759, r5774;
}
{
add.f16x2 r5780, r5685, r5686;
}
{
mul.f16x2 r5783, r5780, r5678;
}
{
add.f16x2 r5786, r5688, r5783;
}
{
add.f16x2 r5789, r5691, r5692;
}
{
mul.f16x2 r5792, r5789, r5680;
}
{
add.f16x2 r5795, r5786, r5792;
}
{
sub.f16x2 r5798, r5697, r5698;
}
{
mul.f16x2 r5801, r5798, r5679;
}
{
sub.f16x2 r5804, r5703, r5704;
}
{
mul.f16x2 r5807, r5804, r5682;
}
{
add.f16x2 r5810, r5801, r5807;
}
{
sub.f16x2 r5813, r5795, r5810;
}
{
add.f16x2 r5816, r5685, r5686;
}
{
mul.f16x2 r5819, r5816, r5678;
}
{
add.f16x2 r5822, r5688, r5819;
}
{
add.f16x2 r5825, r5691, r5692;
}
{
mul.f16x2 r5828, r5825, r5680;
}
{
add.f16x2 r5831, r5822, r5828;
}
{
sub.f16x2 r5834, r5697, r5698;
}
{
mul.f16x2 r5837, r5834, r5679;
}
{
sub.f16x2 r5840, r5703, r5704;
}
{
mul.f16x2 r5843, r5840, r5682;
}
{
add.f16x2 r5846, r5837, r5843;
}
{
add.f16x2 r5849, r5831, r5846;
}
{
add.f16x2 r5852, r5697, r5698;
}
{
mul.f16x2 r5855, r5852, r5676;
}
{
add.f16x2 r5858, r5700, r5855;
}
{
add.f16x2 r5861, r5703, r5704;
}
{
mul.f16x2 r5864, r5861, r5678;
}
{
add.f16x2 r5867, r5858, r5864;
}
{
sub.f16x2 r5870, r5685, r5686;
}
{
mul.f16x2 r5873, r5870, r5677;
}
{
sub.f16x2 r5876, r5691, r5692;
}
{
mul.f16x2 r5879, r5876, r5679;
}
{
add.f16x2 r5882, r5873, r5879;
}
{
add.f16x2 r5885, r5867, r5882;
}
{
add.f16x2 r5888, r5697, r5698;
}
{
mul.f16x2 r5891, r5888, r5676;
}
{
add.f16x2 r5894, r5700, r5891;
}
{
add.f16x2 r5897, r5703, r5704;
}
{
mul.f16x2 r5900, r5897, r5678;
}
{
add.f16x2 r5903, r5894, r5900;
}
{
sub.f16x2 r5906, r5685, r5686;
}
{
mul.f16x2 r5909, r5906, r5677;
}
{
sub.f16x2 r5912, r5691, r5692;
}
{
mul.f16x2 r5915, r5912, r5679;
}
{
add.f16x2 r5918, r5909, r5915;
}
{
sub.f16x2 r5921, r5903, r5918;
}
{
add.f16x2 r5924, r5697, r5698;
}
{
mul.f16x2 r5927, r5924, r5678;
}
{
add.f16x2 r5930, r5700, r5927;
}
{
add.f16x2 r5933, r5703, r5704;
}
{
mul.f16x2 r5936, r5933, r5680;
}
{
add.f16x2 r5939, r5930, r5936;
}
{
sub.f16x2 r5942, r5685, r5686;
}
{
mul.f16x2 r5945, r5942, r5679;
}
{
sub.f16x2 r5948, r5691, r5692;
}
{
mul.f16x2 r5951, r5948, r5682;
}
{
add.f16x2 r5954, r5945, r5951;
}
{
add.f16x2 r5957, r5939, r5954;
}
{
add.f16x2 r5960, r5697, r5698;
}
{
mul.f16x2 r5963, r5960, r5678;
}
{
add.f16x2 r5966, r5700, r5963;
}
{
add.f16x2 r5969, r5703, r5704;
}
{
mul.f16x2 r5972, r5969, r5680;
}
{
add.f16x2 r5975, r5966, r5972;
}
{
sub.f16x2 r5978, r5685, r5686;
}
{
mul.f16x2 r5981, r5978, r5679;
}
{
sub.f16x2 r5984, r5691, r5692;
}
{
mul.f16x2 r5987, r5984, r5682;
}
{
add.f16x2 r5990, r5981, r5987;
}
{
sub.f16x2 r5993, r5975, r5990;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f330;
cvt.rn.f16.f32 high, f330;
mov.b32 r5996, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f332;
cvt.rn.f16.f32 high, f332;
mov.b32 r5997, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f334;
cvt.rn.f16.f32 high, f334;
mov.b32 r5998, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f336;
cvt.rn.f16.f32 high, f336;
mov.b32 r5999, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f338;
cvt.rn.f16.f32 high, f338;
mov.b32 r6000, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f340;
cvt.rn.f16.f32 high, f340;
mov.b32 r6001, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f342;
cvt.rn.f16.f32 high, f342;
mov.b32 r6002, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f344;
cvt.rn.f16.f32 high, f344;
mov.b32 r6003, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f350;
cvt.rn.f16.f32 high, f350;
mov.b32 r6006, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f352;
cvt.rn.f16.f32 high, f352;
mov.b32 r6007, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f358;
cvt.rn.f16.f32 high, f358;
mov.b32 r6010, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f360;
cvt.rn.f16.f32 high, f360;
mov.b32 r6011, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6012, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f364;
cvt.rn.f16.f32 high, f364;
mov.b32 r6013, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f374;
cvt.rn.f16.f32 high, f374;
mov.b32 r6018, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f376;
cvt.rn.f16.f32 high, f376;
mov.b32 r6019, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f390;
cvt.rn.f16.f32 high, f390;
mov.b32 r6026, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f392;
cvt.rn.f16.f32 high, f392;
mov.b32 r6027, {low, high};
}
{
mul.f16x2 r6044, r4781, r5996;
}
{
mul.f16x2 r6047, r4925, r5997;
}
{
sub.f16x2 r6050, r6044, r6047;
}
{
mul.f16x2 r6053, r4781, r5997;
}
{
fma.rn.f16x2 r6056, r4925, r5996, r6053;
}
{
mul.f16x2 r6060, r5101, r5998;
}
{
mul.f16x2 r6063, r5245, r5999;
}
{
sub.f16x2 r6066, r6060, r6063;
}
{
mul.f16x2 r6069, r5101, r5999;
}
{
fma.rn.f16x2 r6072, r5245, r5998, r6069;
}
{
mul.f16x2 r6076, r5421, r6000;
}
{
mul.f16x2 r6079, r5565, r6001;
}
{
sub.f16x2 r6082, r6076, r6079;
}
{
mul.f16x2 r6085, r5421, r6001;
}
{
fma.rn.f16x2 r6088, r5565, r6000, r6085;
}
{
mul.f16x2 r6092, r5741, r6002;
}
{
mul.f16x2 r6095, r5885, r6003;
}
{
sub.f16x2 r6098, r6092, r6095;
}
{
mul.f16x2 r6101, r5741, r6003;
}
{
fma.rn.f16x2 r6104, r5885, r6002, r6101;
}
{
mul.f16x2 r6108, r4853, r5998;
}
{
mul.f16x2 r6111, r4997, r5999;
}
{
sub.f16x2 r6114, r6108, r6111;
}
{
mul.f16x2 r6117, r4853, r5999;
}
{
fma.rn.f16x2 r6120, r4997, r5998, r6117;
}
{
mul.f16x2 r6124, r5173, r6002;
}
{
mul.f16x2 r6127, r5317, r6003;
}
{
sub.f16x2 r6130, r6124, r6127;
}
{
mul.f16x2 r6133, r5173, r6003;
}
{
fma.rn.f16x2 r6136, r5317, r6002, r6133;
}
{
mul.f16x2 r6140, r5493, r6006;
}
{
mul.f16x2 r6143, r5637, r6007;
}
{
sub.f16x2 r6146, r6140, r6143;
}
{
mul.f16x2 r6149, r5493, r6007;
}
{
fma.rn.f16x2 r6152, r5637, r6006, r6149;
}
{
mul.f16x2 r6156, r5813, r6010;
}
{
mul.f16x2 r6159, r5957, r6011;
}
{
sub.f16x2 r6162, r6156, r6159;
}
{
mul.f16x2 r6165, r5813, r6011;
}
{
fma.rn.f16x2 r6168, r5957, r6010, r6165;
}
{
mul.f16x2 r6172, r4889, r6000;
}
{
mul.f16x2 r6175, r5033, r6001;
}
{
sub.f16x2 r6178, r6172, r6175;
}
{
mul.f16x2 r6181, r4889, r6001;
}
{
fma.rn.f16x2 r6184, r5033, r6000, r6181;
}
{
mul.f16x2 r6188, r5209, r6006;
}
{
mul.f16x2 r6191, r5353, r6007;
}
{
sub.f16x2 r6194, r6188, r6191;
}
{
mul.f16x2 r6197, r5209, r6007;
}
{
fma.rn.f16x2 r6200, r5353, r6006, r6197;
}
{
mul.f16x2 r6204, r5529, r6012;
}
{
mul.f16x2 r6207, r5673, r6013;
}
{
sub.f16x2 r6210, r6204, r6207;
}
{
mul.f16x2 r6213, r5529, r6013;
}
{
fma.rn.f16x2 r6216, r5673, r6012, r6213;
}
{
mul.f16x2 r6220, r5849, r6018;
}
{
mul.f16x2 r6223, r5993, r6019;
}
{
sub.f16x2 r6226, r6220, r6223;
}
{
mul.f16x2 r6229, r5849, r6019;
}
{
fma.rn.f16x2 r6232, r5993, r6018, r6229;
}
{
mul.f16x2 r6236, r4817, r6002;
}
{
mul.f16x2 r6239, r4961, r6003;
}
{
sub.f16x2 r6242, r6236, r6239;
}
{
mul.f16x2 r6245, r4817, r6003;
}
{
fma.rn.f16x2 r6248, r4961, r6002, r6245;
}
{
mul.f16x2 r6252, r5137, r6010;
}
{
mul.f16x2 r6255, r5281, r6011;
}
{
sub.f16x2 r6258, r6252, r6255;
}
{
mul.f16x2 r6261, r5137, r6011;
}
{
fma.rn.f16x2 r6264, r5281, r6010, r6261;
}
{
mul.f16x2 r6268, r5457, r6018;
}
{
mul.f16x2 r6271, r5601, r6019;
}
{
sub.f16x2 r6274, r6268, r6271;
}
{
mul.f16x2 r6277, r5457, r6019;
}
{
fma.rn.f16x2 r6280, r5601, r6018, r6277;
}
{
mul.f16x2 r6284, r5777, r6026;
}
{
mul.f16x2 r6287, r5921, r6027;
}
{
sub.f16x2 r6290, r6284, r6287;
}
{
mul.f16x2 r6293, r5777, r6027;
}
{
fma.rn.f16x2 r6296, r5921, r6026, r6293;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6302, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6305, {low, high};
}
{
neg.f16x2 r6306, r6305;
}
{
add.f16x2 r6308, r4733, r5693;
}
{
add.f16x2 r6311, r4413, r6308;
}
{
add.f16x2 r6314, r5053, r5373;
}
{
add.f16x2 r6317, r6311, r6314;
}
{
add.f16x2 r6320, r4745, r5705;
}
{
add.f16x2 r6323, r4425, r6320;
}
{
add.f16x2 r6326, r5065, r5385;
}
{
add.f16x2 r6329, r6323, r6326;
}
{
add.f16x2 r6332, r4733, r5693;
}
{
mul.f16x2 r6335, r6332, r6300;
}
{
add.f16x2 r6338, r4413, r6335;
}
{
add.f16x2 r6341, r5053, r5373;
}
{
mul.f16x2 r6344, r6341, r6302;
}
{
add.f16x2 r6347, r6338, r6344;
}
{
sub.f16x2 r6350, r4745, r5705;
}
{
mul.f16x2 r6353, r6350, r6301;
}
{
sub.f16x2 r6356, r5065, r5385;
}
{
mul.f16x2 r6359, r6356, r6303;
}
{
add.f16x2 r6362, r6353, r6359;
}
{
sub.f16x2 r6365, r6347, r6362;
}
{
add.f16x2 r6368, r4733, r5693;
}
{
mul.f16x2 r6371, r6368, r6300;
}
{
add.f16x2 r6374, r4413, r6371;
}
{
add.f16x2 r6377, r5053, r5373;
}
{
mul.f16x2 r6380, r6377, r6302;
}
{
add.f16x2 r6383, r6374, r6380;
}
{
sub.f16x2 r6386, r4745, r5705;
}
{
mul.f16x2 r6389, r6386, r6301;
}
{
sub.f16x2 r6392, r5065, r5385;
}
{
mul.f16x2 r6395, r6392, r6303;
}
{
add.f16x2 r6398, r6389, r6395;
}
{
add.f16x2 r6401, r6383, r6398;
}
{
add.f16x2 r6404, r4733, r5693;
}
{
mul.f16x2 r6407, r6404, r6302;
}
{
add.f16x2 r6410, r4413, r6407;
}
{
add.f16x2 r6413, r5053, r5373;
}
{
mul.f16x2 r6416, r6413, r6304;
}
{
add.f16x2 r6419, r6410, r6416;
}
{
sub.f16x2 r6422, r4745, r5705;
}
{
mul.f16x2 r6425, r6422, r6303;
}
{
sub.f16x2 r6428, r5065, r5385;
}
{
mul.f16x2 r6431, r6428, r6306;
}
{
add.f16x2 r6434, r6425, r6431;
}
{
sub.f16x2 r6437, r6419, r6434;
}
{
add.f16x2 r6440, r4733, r5693;
}
{
mul.f16x2 r6443, r6440, r6302;
}
{
add.f16x2 r6446, r4413, r6443;
}
{
add.f16x2 r6449, r5053, r5373;
}
{
mul.f16x2 r6452, r6449, r6304;
}
{
add.f16x2 r6455, r6446, r6452;
}
{
sub.f16x2 r6458, r4745, r5705;
}
{
mul.f16x2 r6461, r6458, r6303;
}
{
sub.f16x2 r6464, r5065, r5385;
}
{
mul.f16x2 r6467, r6464, r6306;
}
{
add.f16x2 r6470, r6461, r6467;
}
{
add.f16x2 r6473, r6455, r6470;
}
{
add.f16x2 r6476, r4745, r5705;
}
{
mul.f16x2 r6479, r6476, r6300;
}
{
add.f16x2 r6482, r4425, r6479;
}
{
add.f16x2 r6485, r5065, r5385;
}
{
mul.f16x2 r6488, r6485, r6302;
}
{
add.f16x2 r6491, r6482, r6488;
}
{
sub.f16x2 r6494, r4733, r5693;
}
{
mul.f16x2 r6497, r6494, r6301;
}
{
sub.f16x2 r6500, r5053, r5373;
}
{
mul.f16x2 r6503, r6500, r6303;
}
{
add.f16x2 r6506, r6497, r6503;
}
{
add.f16x2 r6509, r6491, r6506;
}
{
add.f16x2 r6512, r4745, r5705;
}
{
mul.f16x2 r6515, r6512, r6300;
}
{
add.f16x2 r6518, r4425, r6515;
}
{
add.f16x2 r6521, r5065, r5385;
}
{
mul.f16x2 r6524, r6521, r6302;
}
{
add.f16x2 r6527, r6518, r6524;
}
{
sub.f16x2 r6530, r4733, r5693;
}
{
mul.f16x2 r6533, r6530, r6301;
}
{
sub.f16x2 r6536, r5053, r5373;
}
{
mul.f16x2 r6539, r6536, r6303;
}
{
add.f16x2 r6542, r6533, r6539;
}
{
sub.f16x2 r6545, r6527, r6542;
}
{
add.f16x2 r6548, r4745, r5705;
}
{
mul.f16x2 r6551, r6548, r6302;
}
{
add.f16x2 r6554, r4425, r6551;
}
{
add.f16x2 r6557, r5065, r5385;
}
{
mul.f16x2 r6560, r6557, r6304;
}
{
add.f16x2 r6563, r6554, r6560;
}
{
sub.f16x2 r6566, r4733, r5693;
}
{
mul.f16x2 r6569, r6566, r6303;
}
{
sub.f16x2 r6572, r5053, r5373;
}
{
mul.f16x2 r6575, r6572, r6306;
}
{
add.f16x2 r6578, r6569, r6575;
}
{
add.f16x2 r6581, r6563, r6578;
}
{
add.f16x2 r6584, r4745, r5705;
}
{
mul.f16x2 r6587, r6584, r6302;
}
{
add.f16x2 r6590, r4425, r6587;
}
{
add.f16x2 r6593, r5065, r5385;
}
{
mul.f16x2 r6596, r6593, r6304;
}
{
add.f16x2 r6599, r6590, r6596;
}
{
sub.f16x2 r6602, r4733, r5693;
}
{
mul.f16x2 r6605, r6602, r6303;
}
{
sub.f16x2 r6608, r5053, r5373;
}
{
mul.f16x2 r6611, r6608, r6306;
}
{
add.f16x2 r6614, r6605, r6611;
}
{
sub.f16x2 r6617, r6599, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6623, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6625, {low, high};
}
{
neg.f16x2 r6626, r6625;
}
{
add.f16x2 r6628, r6050, r6098;
}
{
add.f16x2 r6631, r4461, r6628;
}
{
add.f16x2 r6634, r6066, r6082;
}
{
add.f16x2 r6637, r6631, r6634;
}
{
add.f16x2 r6640, r6056, r6104;
}
{
add.f16x2 r6643, r4605, r6640;
}
{
add.f16x2 r6646, r6072, r6088;
}
{
add.f16x2 r6649, r6643, r6646;
}
{
add.f16x2 r6652, r6050, r6098;
}
{
mul.f16x2 r6655, r6652, r6620;
}
{
add.f16x2 r6658, r4461, r6655;
}
{
add.f16x2 r6661, r6066, r6082;
}
{
mul.f16x2 r6664, r6661, r6622;
}
{
add.f16x2 r6667, r6658, r6664;
}
{
sub.f16x2 r6670, r6056, r6104;
}
{
mul.f16x2 r6673, r6670, r6621;
}
{
sub.f16x2 r6676, r6072, r6088;
}
{
mul.f16x2 r6679, r6676, r6623;
}
{
add.f16x2 r6682, r6673, r6679;
}
{
sub.f16x2 r6685, r6667, r6682;
}
{
add.f16x2 r6688, r6050, r6098;
}
{
mul.f16x2 r6691, r6688, r6620;
}
{
add.f16x2 r6694, r4461, r6691;
}
{
add.f16x2 r6697, r6066, r6082;
}
{
mul.f16x2 r6700, r6697, r6622;
}
{
add.f16x2 r6703, r6694, r6700;
}
{
sub.f16x2 r6706, r6056, r6104;
}
{
mul.f16x2 r6709, r6706, r6621;
}
{
sub.f16x2 r6712, r6072, r6088;
}
{
mul.f16x2 r6715, r6712, r6623;
}
{
add.f16x2 r6718, r6709, r6715;
}
{
add.f16x2 r6721, r6703, r6718;
}
{
add.f16x2 r6724, r6050, r6098;
}
{
mul.f16x2 r6727, r6724, r6622;
}
{
add.f16x2 r6730, r4461, r6727;
}
{
add.f16x2 r6733, r6066, r6082;
}
{
mul.f16x2 r6736, r6733, r6624;
}
{
add.f16x2 r6739, r6730, r6736;
}
{
sub.f16x2 r6742, r6056, r6104;
}
{
mul.f16x2 r6745, r6742, r6623;
}
{
sub.f16x2 r6748, r6072, r6088;
}
{
mul.f16x2 r6751, r6748, r6626;
}
{
add.f16x2 r6754, r6745, r6751;
}
{
sub.f16x2 r6757, r6739, r6754;
}
{
add.f16x2 r6760, r6050, r6098;
}
{
mul.f16x2 r6763, r6760, r6622;
}
{
add.f16x2 r6766, r4461, r6763;
}
{
add.f16x2 r6769, r6066, r6082;
}
{
mul.f16x2 r6772, r6769, r6624;
}
{
add.f16x2 r6775, r6766, r6772;
}
{
sub.f16x2 r6778, r6056, r6104;
}
{
mul.f16x2 r6781, r6778, r6623;
}
{
sub.f16x2 r6784, r6072, r6088;
}
{
mul.f16x2 r6787, r6784, r6626;
}
{
add.f16x2 r6790, r6781, r6787;
}
{
add.f16x2 r6793, r6775, r6790;
}
{
add.f16x2 r6796, r6056, r6104;
}
{
mul.f16x2 r6799, r6796, r6620;
}
{
add.f16x2 r6802, r4605, r6799;
}
{
add.f16x2 r6805, r6072, r6088;
}
{
mul.f16x2 r6808, r6805, r6622;
}
{
add.f16x2 r6811, r6802, r6808;
}
{
sub.f16x2 r6814, r6050, r6098;
}
{
mul.f16x2 r6817, r6814, r6621;
}
{
sub.f16x2 r6820, r6066, r6082;
}
{
mul.f16x2 r6823, r6820, r6623;
}
{
add.f16x2 r6826, r6817, r6823;
}
{
add.f16x2 r6829, r6811, r6826;
}
{
add.f16x2 r6832, r6056, r6104;
}
{
mul.f16x2 r6835, r6832, r6620;
}
{
add.f16x2 r6838, r4605, r6835;
}
{
add.f16x2 r6841, r6072, r6088;
}
{
mul.f16x2 r6844, r6841, r6622;
}
{
add.f16x2 r6847, r6838, r6844;
}
{
sub.f16x2 r6850, r6050, r6098;
}
{
mul.f16x2 r6853, r6850, r6621;
}
{
sub.f16x2 r6856, r6066, r6082;
}
{
mul.f16x2 r6859, r6856, r6623;
}
{
add.f16x2 r6862, r6853, r6859;
}
{
sub.f16x2 r6865, r6847, r6862;
}
{
add.f16x2 r6868, r6056, r6104;
}
{
mul.f16x2 r6871, r6868, r6622;
}
{
add.f16x2 r6874, r4605, r6871;
}
{
add.f16x2 r6877, r6072, r6088;
}
{
mul.f16x2 r6880, r6877, r6624;
}
{
add.f16x2 r6883, r6874, r6880;
}
{
sub.f16x2 r6886, r6050, r6098;
}
{
mul.f16x2 r6889, r6886, r6623;
}
{
sub.f16x2 r6892, r6066, r6082;
}
{
mul.f16x2 r6895, r6892, r6626;
}
{
add.f16x2 r6898, r6889, r6895;
}
{
add.f16x2 r6901, r6883, r6898;
}
{
add.f16x2 r6904, r6056, r6104;
}
{
mul.f16x2 r6907, r6904, r6622;
}
{
add.f16x2 r6910, r4605, r6907;
}
{
add.f16x2 r6913, r6072, r6088;
}
{
mul.f16x2 r6916, r6913, r6624;
}
{
add.f16x2 r6919, r6910, r6916;
}
{
sub.f16x2 r6922, r6050, r6098;
}
{
mul.f16x2 r6925, r6922, r6623;
}
{
sub.f16x2 r6928, r6066, r6082;
}
{
mul.f16x2 r6931, r6928, r6626;
}
{
add.f16x2 r6934, r6925, r6931;
}
{
sub.f16x2 r6937, r6919, r6934;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r6942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r6943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r6944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r6945, {low, high};
}
{
neg.f16x2 r6946, r6945;
}
{
add.f16x2 r6948, r6114, r6162;
}
{
add.f16x2 r6951, r4533, r6948;
}
{
add.f16x2 r6954, r6130, r6146;
}
{
add.f16x2 r6957, r6951, r6954;
}
{
add.f16x2 r6960, r6120, r6168;
}
{
add.f16x2 r6963, r4677, r6960;
}
{
add.f16x2 r6966, r6136, r6152;
}
{
add.f16x2 r6969, r6963, r6966;
}
{
add.f16x2 r6972, r6114, r6162;
}
{
mul.f16x2 r6975, r6972, r6940;
}
{
add.f16x2 r6978, r4533, r6975;
}
{
add.f16x2 r6981, r6130, r6146;
}
{
mul.f16x2 r6984, r6981, r6942;
}
{
add.f16x2 r6987, r6978, r6984;
}
{
sub.f16x2 r6990, r6120, r6168;
}
{
mul.f16x2 r6993, r6990, r6941;
}
{
sub.f16x2 r6996, r6136, r6152;
}
{
mul.f16x2 r6999, r6996, r6943;
}
{
add.f16x2 r7002, r6993, r6999;
}
{
sub.f16x2 r7005, r6987, r7002;
}
{
add.f16x2 r7008, r6114, r6162;
}
{
mul.f16x2 r7011, r7008, r6940;
}
{
add.f16x2 r7014, r4533, r7011;
}
{
add.f16x2 r7017, r6130, r6146;
}
{
mul.f16x2 r7020, r7017, r6942;
}
{
add.f16x2 r7023, r7014, r7020;
}
{
sub.f16x2 r7026, r6120, r6168;
}
{
mul.f16x2 r7029, r7026, r6941;
}
{
sub.f16x2 r7032, r6136, r6152;
}
{
mul.f16x2 r7035, r7032, r6943;
}
{
add.f16x2 r7038, r7029, r7035;
}
{
add.f16x2 r7041, r7023, r7038;
}
{
add.f16x2 r7044, r6114, r6162;
}
{
mul.f16x2 r7047, r7044, r6942;
}
{
add.f16x2 r7050, r4533, r7047;
}
{
add.f16x2 r7053, r6130, r6146;
}
{
mul.f16x2 r7056, r7053, r6944;
}
{
add.f16x2 r7059, r7050, r7056;
}
{
sub.f16x2 r7062, r6120, r6168;
}
{
mul.f16x2 r7065, r7062, r6943;
}
{
sub.f16x2 r7068, r6136, r6152;
}
{
mul.f16x2 r7071, r7068, r6946;
}
{
add.f16x2 r7074, r7065, r7071;
}
{
sub.f16x2 r7077, r7059, r7074;
}
{
add.f16x2 r7080, r6114, r6162;
}
{
mul.f16x2 r7083, r7080, r6942;
}
{
add.f16x2 r7086, r4533, r7083;
}
{
add.f16x2 r7089, r6130, r6146;
}
{
mul.f16x2 r7092, r7089, r6944;
}
{
add.f16x2 r7095, r7086, r7092;
}
{
sub.f16x2 r7098, r6120, r6168;
}
{
mul.f16x2 r7101, r7098, r6943;
}
{
sub.f16x2 r7104, r6136, r6152;
}
{
mul.f16x2 r7107, r7104, r6946;
}
{
add.f16x2 r7110, r7101, r7107;
}
{
add.f16x2 r7113, r7095, r7110;
}
{
add.f16x2 r7116, r6120, r6168;
}
{
mul.f16x2 r7119, r7116, r6940;
}
{
add.f16x2 r7122, r4677, r7119;
}
{
add.f16x2 r7125, r6136, r6152;
}
{
mul.f16x2 r7128, r7125, r6942;
}
{
add.f16x2 r7131, r7122, r7128;
}
{
sub.f16x2 r7134, r6114, r6162;
}
{
mul.f16x2 r7137, r7134, r6941;
}
{
sub.f16x2 r7140, r6130, r6146;
}
{
mul.f16x2 r7143, r7140, r6943;
}
{
add.f16x2 r7146, r7137, r7143;
}
{
add.f16x2 r7149, r7131, r7146;
}
{
add.f16x2 r7152, r6120, r6168;
}
{
mul.f16x2 r7155, r7152, r6940;
}
{
add.f16x2 r7158, r4677, r7155;
}
{
add.f16x2 r7161, r6136, r6152;
}
{
mul.f16x2 r7164, r7161, r6942;
}
{
add.f16x2 r7167, r7158, r7164;
}
{
sub.f16x2 r7170, r6114, r6162;
}
{
mul.f16x2 r7173, r7170, r6941;
}
{
sub.f16x2 r7176, r6130, r6146;
}
{
mul.f16x2 r7179, r7176, r6943;
}
{
add.f16x2 r7182, r7173, r7179;
}
{
sub.f16x2 r7185, r7167, r7182;
}
{
add.f16x2 r7188, r6120, r6168;
}
{
mul.f16x2 r7191, r7188, r6942;
}
{
add.f16x2 r7194, r4677, r7191;
}
{
add.f16x2 r7197, r6136, r6152;
}
{
mul.f16x2 r7200, r7197, r6944;
}
{
add.f16x2 r7203, r7194, r7200;
}
{
sub.f16x2 r7206, r6114, r6162;
}
{
mul.f16x2 r7209, r7206, r6943;
}
{
sub.f16x2 r7212, r6130, r6146;
}
{
mul.f16x2 r7215, r7212, r6946;
}
{
add.f16x2 r7218, r7209, r7215;
}
{
add.f16x2 r7221, r7203, r7218;
}
{
add.f16x2 r7224, r6120, r6168;
}
{
mul.f16x2 r7227, r7224, r6942;
}
{
add.f16x2 r7230, r4677, r7227;
}
{
add.f16x2 r7233, r6136, r6152;
}
{
mul.f16x2 r7236, r7233, r6944;
}
{
add.f16x2 r7239, r7230, r7236;
}
{
sub.f16x2 r7242, r6114, r6162;
}
{
mul.f16x2 r7245, r7242, r6943;
}
{
sub.f16x2 r7248, r6130, r6146;
}
{
mul.f16x2 r7251, r7248, r6946;
}
{
add.f16x2 r7254, r7245, r7251;
}
{
sub.f16x2 r7257, r7239, r7254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7265, {low, high};
}
{
neg.f16x2 r7266, r7265;
}
{
add.f16x2 r7268, r6178, r6226;
}
{
add.f16x2 r7271, r4569, r7268;
}
{
add.f16x2 r7274, r6194, r6210;
}
{
add.f16x2 r7277, r7271, r7274;
}
{
add.f16x2 r7280, r6184, r6232;
}
{
add.f16x2 r7283, r4713, r7280;
}
{
add.f16x2 r7286, r6200, r6216;
}
{
add.f16x2 r7289, r7283, r7286;
}
{
add.f16x2 r7292, r6178, r6226;
}
{
mul.f16x2 r7295, r7292, r7260;
}
{
add.f16x2 r7298, r4569, r7295;
}
{
add.f16x2 r7301, r6194, r6210;
}
{
mul.f16x2 r7304, r7301, r7262;
}
{
add.f16x2 r7307, r7298, r7304;
}
{
sub.f16x2 r7310, r6184, r6232;
}
{
mul.f16x2 r7313, r7310, r7261;
}
{
sub.f16x2 r7316, r6200, r6216;
}
{
mul.f16x2 r7319, r7316, r7263;
}
{
add.f16x2 r7322, r7313, r7319;
}
{
sub.f16x2 r7325, r7307, r7322;
}
{
add.f16x2 r7328, r6178, r6226;
}
{
mul.f16x2 r7331, r7328, r7260;
}
{
add.f16x2 r7334, r4569, r7331;
}
{
add.f16x2 r7337, r6194, r6210;
}
{
mul.f16x2 r7340, r7337, r7262;
}
{
add.f16x2 r7343, r7334, r7340;
}
{
sub.f16x2 r7346, r6184, r6232;
}
{
mul.f16x2 r7349, r7346, r7261;
}
{
sub.f16x2 r7352, r6200, r6216;
}
{
mul.f16x2 r7355, r7352, r7263;
}
{
add.f16x2 r7358, r7349, r7355;
}
{
add.f16x2 r7361, r7343, r7358;
}
{
add.f16x2 r7364, r6178, r6226;
}
{
mul.f16x2 r7367, r7364, r7262;
}
{
add.f16x2 r7370, r4569, r7367;
}
{
add.f16x2 r7373, r6194, r6210;
}
{
mul.f16x2 r7376, r7373, r7264;
}
{
add.f16x2 r7379, r7370, r7376;
}
{
sub.f16x2 r7382, r6184, r6232;
}
{
mul.f16x2 r7385, r7382, r7263;
}
{
sub.f16x2 r7388, r6200, r6216;
}
{
mul.f16x2 r7391, r7388, r7266;
}
{
add.f16x2 r7394, r7385, r7391;
}
{
sub.f16x2 r7397, r7379, r7394;
}
{
add.f16x2 r7400, r6178, r6226;
}
{
mul.f16x2 r7403, r7400, r7262;
}
{
add.f16x2 r7406, r4569, r7403;
}
{
add.f16x2 r7409, r6194, r6210;
}
{
mul.f16x2 r7412, r7409, r7264;
}
{
add.f16x2 r7415, r7406, r7412;
}
{
sub.f16x2 r7418, r6184, r6232;
}
{
mul.f16x2 r7421, r7418, r7263;
}
{
sub.f16x2 r7424, r6200, r6216;
}
{
mul.f16x2 r7427, r7424, r7266;
}
{
add.f16x2 r7430, r7421, r7427;
}
{
add.f16x2 r7433, r7415, r7430;
}
{
add.f16x2 r7436, r6184, r6232;
}
{
mul.f16x2 r7439, r7436, r7260;
}
{
add.f16x2 r7442, r4713, r7439;
}
{
add.f16x2 r7445, r6200, r6216;
}
{
mul.f16x2 r7448, r7445, r7262;
}
{
add.f16x2 r7451, r7442, r7448;
}
{
sub.f16x2 r7454, r6178, r6226;
}
{
mul.f16x2 r7457, r7454, r7261;
}
{
sub.f16x2 r7460, r6194, r6210;
}
{
mul.f16x2 r7463, r7460, r7263;
}
{
add.f16x2 r7466, r7457, r7463;
}
{
add.f16x2 r7469, r7451, r7466;
}
{
add.f16x2 r7472, r6184, r6232;
}
{
mul.f16x2 r7475, r7472, r7260;
}
{
add.f16x2 r7478, r4713, r7475;
}
{
add.f16x2 r7481, r6200, r6216;
}
{
mul.f16x2 r7484, r7481, r7262;
}
{
add.f16x2 r7487, r7478, r7484;
}
{
sub.f16x2 r7490, r6178, r6226;
}
{
mul.f16x2 r7493, r7490, r7261;
}
{
sub.f16x2 r7496, r6194, r6210;
}
{
mul.f16x2 r7499, r7496, r7263;
}
{
add.f16x2 r7502, r7493, r7499;
}
{
sub.f16x2 r7505, r7487, r7502;
}
{
add.f16x2 r7508, r6184, r6232;
}
{
mul.f16x2 r7511, r7508, r7262;
}
{
add.f16x2 r7514, r4713, r7511;
}
{
add.f16x2 r7517, r6200, r6216;
}
{
mul.f16x2 r7520, r7517, r7264;
}
{
add.f16x2 r7523, r7514, r7520;
}
{
sub.f16x2 r7526, r6178, r6226;
}
{
mul.f16x2 r7529, r7526, r7263;
}
{
sub.f16x2 r7532, r6194, r6210;
}
{
mul.f16x2 r7535, r7532, r7266;
}
{
add.f16x2 r7538, r7529, r7535;
}
{
add.f16x2 r7541, r7523, r7538;
}
{
add.f16x2 r7544, r6184, r6232;
}
{
mul.f16x2 r7547, r7544, r7262;
}
{
add.f16x2 r7550, r4713, r7547;
}
{
add.f16x2 r7553, r6200, r6216;
}
{
mul.f16x2 r7556, r7553, r7264;
}
{
add.f16x2 r7559, r7550, r7556;
}
{
sub.f16x2 r7562, r6178, r6226;
}
{
mul.f16x2 r7565, r7562, r7263;
}
{
sub.f16x2 r7568, r6194, r6210;
}
{
mul.f16x2 r7571, r7568, r7266;
}
{
add.f16x2 r7574, r7565, r7571;
}
{
sub.f16x2 r7577, r7559, r7574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r7582, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r7583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r7584, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r7585, {low, high};
}
{
neg.f16x2 r7586, r7585;
}
{
add.f16x2 r7588, r6242, r6290;
}
{
add.f16x2 r7591, r4497, r7588;
}
{
add.f16x2 r7594, r6258, r6274;
}
{
add.f16x2 r7597, r7591, r7594;
}
{
add.f16x2 r7600, r6248, r6296;
}
{
add.f16x2 r7603, r4641, r7600;
}
{
add.f16x2 r7606, r6264, r6280;
}
{
add.f16x2 r7609, r7603, r7606;
}
{
add.f16x2 r7612, r6242, r6290;
}
{
mul.f16x2 r7615, r7612, r7580;
}
{
add.f16x2 r7618, r4497, r7615;
}
{
add.f16x2 r7621, r6258, r6274;
}
{
mul.f16x2 r7624, r7621, r7582;
}
{
add.f16x2 r7627, r7618, r7624;
}
{
sub.f16x2 r7630, r6248, r6296;
}
{
mul.f16x2 r7633, r7630, r7581;
}
{
sub.f16x2 r7636, r6264, r6280;
}
{
mul.f16x2 r7639, r7636, r7583;
}
{
add.f16x2 r7642, r7633, r7639;
}
{
sub.f16x2 r7645, r7627, r7642;
}
{
add.f16x2 r7648, r6242, r6290;
}
{
mul.f16x2 r7651, r7648, r7580;
}
{
add.f16x2 r7654, r4497, r7651;
}
{
add.f16x2 r7657, r6258, r6274;
}
{
mul.f16x2 r7660, r7657, r7582;
}
{
add.f16x2 r7663, r7654, r7660;
}
{
sub.f16x2 r7666, r6248, r6296;
}
{
mul.f16x2 r7669, r7666, r7581;
}
{
sub.f16x2 r7672, r6264, r6280;
}
{
mul.f16x2 r7675, r7672, r7583;
}
{
add.f16x2 r7678, r7669, r7675;
}
{
add.f16x2 r7681, r7663, r7678;
}
{
add.f16x2 r7684, r6242, r6290;
}
{
mul.f16x2 r7687, r7684, r7582;
}
{
add.f16x2 r7690, r4497, r7687;
}
{
add.f16x2 r7693, r6258, r6274;
}
{
mul.f16x2 r7696, r7693, r7584;
}
{
add.f16x2 r7699, r7690, r7696;
}
{
sub.f16x2 r7702, r6248, r6296;
}
{
mul.f16x2 r7705, r7702, r7583;
}
{
sub.f16x2 r7708, r6264, r6280;
}
{
mul.f16x2 r7711, r7708, r7586;
}
{
add.f16x2 r7714, r7705, r7711;
}
{
sub.f16x2 r7717, r7699, r7714;
}
{
add.f16x2 r7720, r6242, r6290;
}
{
mul.f16x2 r7723, r7720, r7582;
}
{
add.f16x2 r7726, r4497, r7723;
}
{
add.f16x2 r7729, r6258, r6274;
}
{
mul.f16x2 r7732, r7729, r7584;
}
{
add.f16x2 r7735, r7726, r7732;
}
{
sub.f16x2 r7738, r6248, r6296;
}
{
mul.f16x2 r7741, r7738, r7583;
}
{
sub.f16x2 r7744, r6264, r6280;
}
{
mul.f16x2 r7747, r7744, r7586;
}
{
add.f16x2 r7750, r7741, r7747;
}
{
add.f16x2 r7753, r7735, r7750;
}
{
add.f16x2 r7756, r6248, r6296;
}
{
mul.f16x2 r7759, r7756, r7580;
}
{
add.f16x2 r7762, r4641, r7759;
}
{
add.f16x2 r7765, r6264, r6280;
}
{
mul.f16x2 r7768, r7765, r7582;
}
{
add.f16x2 r7771, r7762, r7768;
}
{
sub.f16x2 r7774, r6242, r6290;
}
{
mul.f16x2 r7777, r7774, r7581;
}
{
sub.f16x2 r7780, r6258, r6274;
}
{
mul.f16x2 r7783, r7780, r7583;
}
{
add.f16x2 r7786, r7777, r7783;
}
{
add.f16x2 r7789, r7771, r7786;
}
{
add.f16x2 r7792, r6248, r6296;
}
{
mul.f16x2 r7795, r7792, r7580;
}
{
add.f16x2 r7798, r4641, r7795;
}
{
add.f16x2 r7801, r6264, r6280;
}
{
mul.f16x2 r7804, r7801, r7582;
}
{
add.f16x2 r7807, r7798, r7804;
}
{
sub.f16x2 r7810, r6242, r6290;
}
{
mul.f16x2 r7813, r7810, r7581;
}
{
sub.f16x2 r7816, r6258, r6274;
}
{
mul.f16x2 r7819, r7816, r7583;
}
{
add.f16x2 r7822, r7813, r7819;
}
{
sub.f16x2 r7825, r7807, r7822;
}
{
add.f16x2 r7828, r6248, r6296;
}
{
mul.f16x2 r7831, r7828, r7582;
}
{
add.f16x2 r7834, r4641, r7831;
}
{
add.f16x2 r7837, r6264, r6280;
}
{
mul.f16x2 r7840, r7837, r7584;
}
{
add.f16x2 r7843, r7834, r7840;
}
{
sub.f16x2 r7846, r6242, r6290;
}
{
mul.f16x2 r7849, r7846, r7583;
}
{
sub.f16x2 r7852, r6258, r6274;
}
{
mul.f16x2 r7855, r7852, r7586;
}
{
add.f16x2 r7858, r7849, r7855;
}
{
add.f16x2 r7861, r7843, r7858;
}
{
add.f16x2 r7864, r6248, r6296;
}
{
mul.f16x2 r7867, r7864, r7582;
}
{
add.f16x2 r7870, r4641, r7867;
}
{
add.f16x2 r7873, r6264, r6280;
}
{
mul.f16x2 r7876, r7873, r7584;
}
{
add.f16x2 r7879, r7870, r7876;
}
{
sub.f16x2 r7882, r6242, r6290;
}
{
mul.f16x2 r7885, r7882, r7583;
}
{
sub.f16x2 r7888, r6258, r6274;
}
{
mul.f16x2 r7891, r7888, r7586;
}
{
add.f16x2 r7894, r7885, r7891;
}
{
sub.f16x2 r7897, r7879, r7894;
}
mul.wide.u32 rd4, r10397, 1374389535;
shr.u64 rd5, rd4, 35;
cvt.u32.u64 r10401, rd5;
mul.lo.s32 r10402, r10401, 25;
sub.s32 r10403, r10397, r10402;
shl.b32 r10404, r10403, 2;
add.s32 r10405, r10398, r10404;
cvt.rn.f32.u32 f600, r10401;
mul.f32 f601, f600, 0f3D4DE32E;
cos.approx.f32 f485, f601;
sin.approx.f32 f602, f601;
neg.f32 f486, f602;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f485;
cvt.rn.f16.f32 high, f486;
mov.b32 r7900, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7903, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7905, {high, high};
}
{
mul.f16x2 r7907, r6649, r7905;
}
{
neg.f16x2 r7910, r7907;
}
{
fma.rn.f16x2 r7912, r6637, r7903, r7910;
}
{
mul.f16x2 r7916, r6637, r7905;
}
{
fma.rn.f16x2 r7919, r6649, r7903, r7916;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7923, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7925, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7927, {low, high};
}
{
mul.f16x2 r7928, r7925, r7927;
}
{
mul.f16x2 r7931, r7900, r7923;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7934, {high, low};
}
{
fma.rn.f16x2 r7936, r7928, r7934, r7931;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7940, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7942, {high, high};
}
{
mul.f16x2 r7944, r6969, r7942;
}
{
neg.f16x2 r7947, r7944;
}
{
fma.rn.f16x2 r7949, r6957, r7940, r7947;
}
{
mul.f16x2 r7953, r6957, r7942;
}
{
fma.rn.f16x2 r7956, r6969, r7940, r7953;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7960, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7962, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r7964, {low, high};
}
{
mul.f16x2 r7965, r7962, r7964;
}
{
mul.f16x2 r7968, r7936, r7960;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7936;
mov.b32 r7971, {high, low};
}
{
fma.rn.f16x2 r7973, r7965, r7971, r7968;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r7977, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r7979, {high, high};
}
{
mul.f16x2 r7981, r7289, r7979;
}
{
neg.f16x2 r7984, r7981;
}
{
fma.rn.f16x2 r7986, r7277, r7977, r7984;
}
{
mul.f16x2 r7990, r7277, r7979;
}
{
fma.rn.f16x2 r7993, r7289, r7977, r7990;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7997, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r7999, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8001, {low, high};
}
{
mul.f16x2 r8002, r7999, r8001;
}
{
mul.f16x2 r8005, r7973, r7997;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7973;
mov.b32 r8008, {high, low};
}
{
fma.rn.f16x2 r8010, r8002, r8008, r8005;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8014, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8016, {high, high};
}
{
mul.f16x2 r8018, r7609, r8016;
}
{
neg.f16x2 r8021, r8018;
}
{
fma.rn.f16x2 r8023, r7597, r8014, r8021;
}
{
mul.f16x2 r8027, r7597, r8016;
}
{
fma.rn.f16x2 r8030, r7609, r8014, r8027;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8034, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8036, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8038, {low, high};
}
{
mul.f16x2 r8039, r8036, r8038;
}
{
mul.f16x2 r8042, r8010, r8034;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8010;
mov.b32 r8045, {high, low};
}
{
fma.rn.f16x2 r8047, r8039, r8045, r8042;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8051, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8053, {high, high};
}
{
mul.f16x2 r8055, r6509, r8053;
}
{
neg.f16x2 r8058, r8055;
}
{
fma.rn.f16x2 r8060, r6365, r8051, r8058;
}
{
mul.f16x2 r8064, r6365, r8053;
}
{
fma.rn.f16x2 r8067, r6509, r8051, r8064;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8071, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8073, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8075, {low, high};
}
{
mul.f16x2 r8076, r8073, r8075;
}
{
mul.f16x2 r8079, r8047, r8071;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8047;
mov.b32 r8082, {high, low};
}
{
fma.rn.f16x2 r8084, r8076, r8082, r8079;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8088, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8090, {high, high};
}
{
mul.f16x2 r8092, r6829, r8090;
}
{
neg.f16x2 r8095, r8092;
}
{
fma.rn.f16x2 r8097, r6685, r8088, r8095;
}
{
mul.f16x2 r8101, r6685, r8090;
}
{
fma.rn.f16x2 r8104, r6829, r8088, r8101;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8108, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8110, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8112, {low, high};
}
{
mul.f16x2 r8113, r8110, r8112;
}
{
mul.f16x2 r8116, r8084, r8108;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8084;
mov.b32 r8119, {high, low};
}
{
fma.rn.f16x2 r8121, r8113, r8119, r8116;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8125, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8127, {high, high};
}
{
mul.f16x2 r8129, r7149, r8127;
}
{
neg.f16x2 r8132, r8129;
}
{
fma.rn.f16x2 r8134, r7005, r8125, r8132;
}
{
mul.f16x2 r8138, r7005, r8127;
}
{
fma.rn.f16x2 r8141, r7149, r8125, r8138;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8145, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8147, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8149, {low, high};
}
{
mul.f16x2 r8150, r8147, r8149;
}
{
mul.f16x2 r8153, r8121, r8145;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8121;
mov.b32 r8156, {high, low};
}
{
fma.rn.f16x2 r8158, r8150, r8156, r8153;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8162, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8164, {high, high};
}
{
mul.f16x2 r8166, r7469, r8164;
}
{
neg.f16x2 r8169, r8166;
}
{
fma.rn.f16x2 r8171, r7325, r8162, r8169;
}
{
mul.f16x2 r8175, r7325, r8164;
}
{
fma.rn.f16x2 r8178, r7469, r8162, r8175;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8182, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8184, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8186, {low, high};
}
{
mul.f16x2 r8187, r8184, r8186;
}
{
mul.f16x2 r8190, r8158, r8182;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8158;
mov.b32 r8193, {high, low};
}
{
fma.rn.f16x2 r8195, r8187, r8193, r8190;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8199, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8201, {high, high};
}
{
mul.f16x2 r8203, r7789, r8201;
}
{
neg.f16x2 r8206, r8203;
}
{
fma.rn.f16x2 r8208, r7645, r8199, r8206;
}
{
mul.f16x2 r8212, r7645, r8201;
}
{
fma.rn.f16x2 r8215, r7789, r8199, r8212;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8219, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8221, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8223, {low, high};
}
{
mul.f16x2 r8224, r8221, r8223;
}
{
mul.f16x2 r8227, r8195, r8219;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8195;
mov.b32 r8230, {high, low};
}
{
fma.rn.f16x2 r8232, r8224, r8230, r8227;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8236, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8238, {high, high};
}
{
mul.f16x2 r8240, r6581, r8238;
}
{
neg.f16x2 r8243, r8240;
}
{
fma.rn.f16x2 r8245, r6437, r8236, r8243;
}
{
mul.f16x2 r8249, r6437, r8238;
}
{
fma.rn.f16x2 r8252, r6581, r8236, r8249;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8256, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8258, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8260, {low, high};
}
{
mul.f16x2 r8261, r8258, r8260;
}
{
mul.f16x2 r8264, r8232, r8256;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8232;
mov.b32 r8267, {high, low};
}
{
fma.rn.f16x2 r8269, r8261, r8267, r8264;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8273, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8275, {high, high};
}
{
mul.f16x2 r8277, r6901, r8275;
}
{
neg.f16x2 r8280, r8277;
}
{
fma.rn.f16x2 r8282, r6757, r8273, r8280;
}
{
mul.f16x2 r8286, r6757, r8275;
}
{
fma.rn.f16x2 r8289, r6901, r8273, r8286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8293, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8295, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8297, {low, high};
}
{
mul.f16x2 r8298, r8295, r8297;
}
{
mul.f16x2 r8301, r8269, r8293;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8269;
mov.b32 r8304, {high, low};
}
{
fma.rn.f16x2 r8306, r8298, r8304, r8301;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8310, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8312, {high, high};
}
{
mul.f16x2 r8314, r7221, r8312;
}
{
neg.f16x2 r8317, r8314;
}
{
fma.rn.f16x2 r8319, r7077, r8310, r8317;
}
{
mul.f16x2 r8323, r7077, r8312;
}
{
fma.rn.f16x2 r8326, r7221, r8310, r8323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8330, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8332, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8334, {low, high};
}
{
mul.f16x2 r8335, r8332, r8334;
}
{
mul.f16x2 r8338, r8306, r8330;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8306;
mov.b32 r8341, {high, low};
}
{
fma.rn.f16x2 r8343, r8335, r8341, r8338;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8347, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8349, {high, high};
}
{
mul.f16x2 r8351, r7541, r8349;
}
{
neg.f16x2 r8354, r8351;
}
{
fma.rn.f16x2 r8356, r7397, r8347, r8354;
}
{
mul.f16x2 r8360, r7397, r8349;
}
{
fma.rn.f16x2 r8363, r7541, r8347, r8360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8367, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8369, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8371, {low, high};
}
{
mul.f16x2 r8372, r8369, r8371;
}
{
mul.f16x2 r8375, r8343, r8367;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8343;
mov.b32 r8378, {high, low};
}
{
fma.rn.f16x2 r8380, r8372, r8378, r8375;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8384, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8386, {high, high};
}
{
mul.f16x2 r8388, r7861, r8386;
}
{
neg.f16x2 r8391, r8388;
}
{
fma.rn.f16x2 r8393, r7717, r8384, r8391;
}
{
mul.f16x2 r8397, r7717, r8386;
}
{
fma.rn.f16x2 r8400, r7861, r8384, r8397;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8404, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8406, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8408, {low, high};
}
{
mul.f16x2 r8409, r8406, r8408;
}
{
mul.f16x2 r8412, r8380, r8404;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8380;
mov.b32 r8415, {high, low};
}
{
fma.rn.f16x2 r8417, r8409, r8415, r8412;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8421, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8423, {high, high};
}
{
mul.f16x2 r8425, r6617, r8423;
}
{
neg.f16x2 r8428, r8425;
}
{
fma.rn.f16x2 r8430, r6473, r8421, r8428;
}
{
mul.f16x2 r8434, r6473, r8423;
}
{
fma.rn.f16x2 r8437, r6617, r8421, r8434;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8441, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8443, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8445, {low, high};
}
{
mul.f16x2 r8446, r8443, r8445;
}
{
mul.f16x2 r8449, r8417, r8441;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8417;
mov.b32 r8452, {high, low};
}
{
fma.rn.f16x2 r8454, r8446, r8452, r8449;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8458, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8460, {high, high};
}
{
mul.f16x2 r8462, r6937, r8460;
}
{
neg.f16x2 r8465, r8462;
}
{
fma.rn.f16x2 r8467, r6793, r8458, r8465;
}
{
mul.f16x2 r8471, r6793, r8460;
}
{
fma.rn.f16x2 r8474, r6937, r8458, r8471;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8478, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8480, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8482, {low, high};
}
{
mul.f16x2 r8483, r8480, r8482;
}
{
mul.f16x2 r8486, r8454, r8478;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8454;
mov.b32 r8489, {high, low};
}
{
fma.rn.f16x2 r8491, r8483, r8489, r8486;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8495, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8497, {high, high};
}
{
mul.f16x2 r8499, r7257, r8497;
}
{
neg.f16x2 r8502, r8499;
}
{
fma.rn.f16x2 r8504, r7113, r8495, r8502;
}
{
mul.f16x2 r8508, r7113, r8497;
}
{
fma.rn.f16x2 r8511, r7257, r8495, r8508;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8515, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8517, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8519, {low, high};
}
{
mul.f16x2 r8520, r8517, r8519;
}
{
mul.f16x2 r8523, r8491, r8515;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8491;
mov.b32 r8526, {high, low};
}
{
fma.rn.f16x2 r8528, r8520, r8526, r8523;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8532, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8534, {high, high};
}
{
mul.f16x2 r8536, r7577, r8534;
}
{
neg.f16x2 r8539, r8536;
}
{
fma.rn.f16x2 r8541, r7433, r8532, r8539;
}
{
mul.f16x2 r8545, r7433, r8534;
}
{
fma.rn.f16x2 r8548, r7577, r8532, r8545;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8552, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8554, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8556, {low, high};
}
{
mul.f16x2 r8557, r8554, r8556;
}
{
mul.f16x2 r8560, r8528, r8552;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8528;
mov.b32 r8563, {high, low};
}
{
fma.rn.f16x2 r8565, r8557, r8563, r8560;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8569, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8571, {high, high};
}
{
mul.f16x2 r8573, r7897, r8571;
}
{
neg.f16x2 r8576, r8573;
}
{
fma.rn.f16x2 r8578, r7753, r8569, r8576;
}
{
mul.f16x2 r8582, r7753, r8571;
}
{
fma.rn.f16x2 r8585, r7897, r8569, r8582;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8589, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8591, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8593, {low, high};
}
{
mul.f16x2 r8594, r8591, r8593;
}
{
mul.f16x2 r8597, r8565, r8589;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8565;
mov.b32 r8600, {high, low};
}
{
fma.rn.f16x2 r8602, r8594, r8600, r8597;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8606, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8608, {high, high};
}
{
mul.f16x2 r8610, r6545, r8608;
}
{
neg.f16x2 r8613, r8610;
}
{
fma.rn.f16x2 r8615, r6401, r8606, r8613;
}
{
mul.f16x2 r8619, r6401, r8608;
}
{
fma.rn.f16x2 r8622, r6545, r8606, r8619;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8626, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8628, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8630, {low, high};
}
{
mul.f16x2 r8631, r8628, r8630;
}
{
mul.f16x2 r8634, r8602, r8626;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8602;
mov.b32 r8637, {high, low};
}
{
fma.rn.f16x2 r8639, r8631, r8637, r8634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8643, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8645, {high, high};
}
{
mul.f16x2 r8647, r6865, r8645;
}
{
neg.f16x2 r8650, r8647;
}
{
fma.rn.f16x2 r8652, r6721, r8643, r8650;
}
{
mul.f16x2 r8656, r6721, r8645;
}
{
fma.rn.f16x2 r8659, r6865, r8643, r8656;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8663, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8665, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8667, {low, high};
}
{
mul.f16x2 r8668, r8665, r8667;
}
{
mul.f16x2 r8671, r8639, r8663;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8639;
mov.b32 r8674, {high, low};
}
{
fma.rn.f16x2 r8676, r8668, r8674, r8671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8680, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8682, {high, high};
}
{
mul.f16x2 r8684, r7185, r8682;
}
{
neg.f16x2 r8687, r8684;
}
{
fma.rn.f16x2 r8689, r7041, r8680, r8687;
}
{
mul.f16x2 r8693, r7041, r8682;
}
{
fma.rn.f16x2 r8696, r7185, r8680, r8693;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8700, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8702, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8704, {low, high};
}
{
mul.f16x2 r8705, r8702, r8704;
}
{
mul.f16x2 r8708, r8676, r8700;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8676;
mov.b32 r8711, {high, low};
}
{
fma.rn.f16x2 r8713, r8705, r8711, r8708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8717, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8719, {high, high};
}
{
mul.f16x2 r8721, r7505, r8719;
}
{
neg.f16x2 r8724, r8721;
}
{
fma.rn.f16x2 r8726, r7361, r8717, r8724;
}
{
mul.f16x2 r8730, r7361, r8719;
}
{
fma.rn.f16x2 r8733, r7505, r8717, r8730;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r7900;
mov.b32 r8739, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f533;
cvt.rn.f16.f32 high, f534;
mov.b32 r8741, {low, high};
}
{
mul.f16x2 r8742, r8739, r8741;
}
{
mul.f16x2 r8745, r8713, r8737;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8713;
mov.b32 r8748, {high, low};
}
{
fma.rn.f16x2 r8750, r8742, r8748, r8745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8750;
mov.b32 r8754, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r8750;
mov.b32 r8756, {high, high};
}
{
mul.f16x2 r8758, r7825, r8756;
}
{
neg.f16x2 r8761, r8758;
}
{
fma.rn.f16x2 r8763, r7681, r8754, r8761;
}
{
mul.f16x2 r8767, r7681, r8756;
}
{
fma.rn.f16x2 r8770, r7825, r8754, r8767;
}
barrier.sync 0;
mad.lo.s32 r10406, r10401, 2500, r10405;
st.shared.u32 [r10406], r6317;
st.shared.u32 [r10406+100], r7912;
st.shared.u32 [r10406+200], r7949;
st.shared.u32 [r10406+300], r7986;
st.shared.u32 [r10406+400], r8023;
st.shared.u32 [r10406+500], r8060;
st.shared.u32 [r10406+600], r8097;
st.shared.u32 [r10406+700], r8134;
st.shared.u32 [r10406+800], r8171;
st.shared.u32 [r10406+900], r8208;
st.shared.u32 [r10406+1000], r8245;
st.shared.u32 [r10406+1100], r8282;
st.shared.u32 [r10406+1200], r8319;
st.shared.u32 [r10406+1300], r8356;
st.shared.u32 [r10406+1400], r8393;
st.shared.u32 [r10406+1500], r8430;
st.shared.u32 [r10406+1600], r8467;
st.shared.u32 [r10406+1700], r8504;
st.shared.u32 [r10406+1800], r8541;
st.shared.u32 [r10406+1900], r8578;
st.shared.u32 [r10406+2000], r8615;
st.shared.u32 [r10406+2100], r8652;
st.shared.u32 [r10406+2200], r8689;
st.shared.u32 [r10406+2300], r8726;
st.shared.u32 [r10406+2400], r8763;
barrier.sync 0;
ld.shared.u32 r8803, [r10400];
ld.shared.u32 r9123, [r10400+500];
ld.shared.u32 r9443, [r10400+1000];
ld.shared.u32 r9763, [r10400+1500];
ld.shared.u32 r10083, [r10400+2000];
ld.shared.u32 r8800, [r10400+2500];
ld.shared.u32 r9120, [r10400+3000];
ld.shared.u32 r9440, [r10400+3500];
ld.shared.u32 r9760, [r10400+4000];
ld.shared.u32 r10080, [r10400+4500];
ld.shared.u32 r8806, [r10400+5000];
ld.shared.u32 r9126, [r10400+5500];
ld.shared.u32 r9446, [r10400+6000];
ld.shared.u32 r9766, [r10400+6500];
ld.shared.u32 r10086, [r10400+7000];
ld.shared.u32 r8807, [r10400+7500];
ld.shared.u32 r9127, [r10400+8000];
ld.shared.u32 r9447, [r10400+8500];
ld.shared.u32 r9767, [r10400+9000];
ld.shared.u32 r10087, [r10400+9500];
ld.shared.u32 r8801, [r10400+10000];
ld.shared.u32 r9121, [r10400+10500];
ld.shared.u32 r9441, [r10400+11000];
ld.shared.u32 r9761, [r10400+11500];
ld.shared.u32 r10081, [r10400+12000];
barrier.sync 0;
st.shared.u32 [r10406], r6329;
st.shared.u32 [r10406+100], r7919;
st.shared.u32 [r10406+200], r7956;
st.shared.u32 [r10406+300], r7993;
st.shared.u32 [r10406+400], r8030;
st.shared.u32 [r10406+500], r8067;
st.shared.u32 [r10406+600], r8104;
st.shared.u32 [r10406+700], r8141;
st.shared.u32 [r10406+800], r8178;
st.shared.u32 [r10406+900], r8215;
st.shared.u32 [r10406+1000], r8252;
st.shared.u32 [r10406+1100], r8289;
st.shared.u32 [r10406+1200], r8326;
st.shared.u32 [r10406+1300], r8363;
st.shared.u32 [r10406+1400], r8400;
st.shared.u32 [r10406+1500], r8437;
st.shared.u32 [r10406+1600], r8474;
st.shared.u32 [r10406+1700], r8511;
st.shared.u32 [r10406+1800], r8548;
st.shared.u32 [r10406+1900], r8585;
st.shared.u32 [r10406+2000], r8622;
st.shared.u32 [r10406+2100], r8659;
st.shared.u32 [r10406+2200], r8696;
st.shared.u32 [r10406+2300], r8733;
st.shared.u32 [r10406+2400], r8770;
barrier.sync 0;
ld.shared.u32 r8815, [r10400];
ld.shared.u32 r9135, [r10400+500];
ld.shared.u32 r9455, [r10400+1000];
ld.shared.u32 r9775, [r10400+1500];
ld.shared.u32 r10095, [r10400+2000];
ld.shared.u32 r8812, [r10400+2500];
ld.shared.u32 r9132, [r10400+3000];
ld.shared.u32 r9452, [r10400+3500];
ld.shared.u32 r9772, [r10400+4000];
ld.shared.u32 r10092, [r10400+4500];
ld.shared.u32 r8818, [r10400+5000];
ld.shared.u32 r9138, [r10400+5500];
ld.shared.u32 r9458, [r10400+6000];
ld.shared.u32 r9778, [r10400+6500];
ld.shared.u32 r10098, [r10400+7000];
ld.shared.u32 r8819, [r10400+7500];
ld.shared.u32 r9139, [r10400+8000];
ld.shared.u32 r9459, [r10400+8500];
ld.shared.u32 r9779, [r10400+9000];
ld.shared.u32 r10099, [r10400+9500];
ld.shared.u32 r8813, [r10400+10000];
ld.shared.u32 r9133, [r10400+10500];
ld.shared.u32 r9453, [r10400+11000];
ld.shared.u32 r9773, [r10400+11500];
ld.shared.u32 r10093, [r10400+12000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8791, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8792, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r8793, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r8794, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r8795, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r8796, {low, high};
}
{
neg.f16x2 r8797, r8796;
}
{
add.f16x2 r8799, r8800, r8801;
}
{
add.f16x2 r8802, r8803, r8799;
}
{
add.f16x2 r8805, r8806, r8807;
}
{
add.f16x2 %0, r8802, r8805;
}
{
add.f16x2 r8811, r8812, r8813;
}
{
add.f16x2 r8814, r8815, r8811;
}
{
add.f16x2 r8817, r8818, r8819;
}
{
add.f16x2 %1, r8814, r8817;
}
{
add.f16x2 r8823, r8800, r8801;
}
{
mul.f16x2 r8826, r8823, r8791;
}
{
add.f16x2 r8829, r8803, r8826;
}
{
add.f16x2 r8832, r8806, r8807;
}
{
mul.f16x2 r8835, r8832, r8793;
}
{
add.f16x2 r8838, r8829, r8835;
}
{
sub.f16x2 r8841, r8812, r8813;
}
{
mul.f16x2 r8844, r8841, r8792;
}
{
sub.f16x2 r8847, r8818, r8819;
}
{
mul.f16x2 r8850, r8847, r8794;
}
{
add.f16x2 r8853, r8844, r8850;
}
{
sub.f16x2 %10, r8838, r8853;
}
{
add.f16x2 r8859, r8800, r8801;
}
{
mul.f16x2 r8862, r8859, r8791;
}
{
add.f16x2 r8865, r8803, r8862;
}
{
add.f16x2 r8868, r8806, r8807;
}
{
mul.f16x2 r8871, r8868, r8793;
}
{
add.f16x2 r8874, r8865, r8871;
}
{
sub.f16x2 r8877, r8812, r8813;
}
{
mul.f16x2 r8880, r8877, r8792;
}
{
sub.f16x2 r8883, r8818, r8819;
}
{
mul.f16x2 r8886, r8883, r8794;
}
{
add.f16x2 r8889, r8880, r8886;
}
{
add.f16x2 %40, r8874, r8889;
}
{
add.f16x2 r8895, r8800, r8801;
}
{
mul.f16x2 r8898, r8895, r8793;
}
{
add.f16x2 r8901, r8803, r8898;
}
{
add.f16x2 r8904, r8806, r8807;
}
{
mul.f16x2 r8907, r8904, r8795;
}
{
add.f16x2 r8910, r8901, r8907;
}
{
sub.f16x2 r8913, r8812, r8813;
}
{
mul.f16x2 r8916, r8913, r8794;
}
{
sub.f16x2 r8919, r8818, r8819;
}
{
mul.f16x2 r8922, r8919, r8797;
}
{
add.f16x2 r8925, r8916, r8922;
}
{
sub.f16x2 %20, r8910, r8925;
}
{
add.f16x2 r8931, r8800, r8801;
}
{
mul.f16x2 r8934, r8931, r8793;
}
{
add.f16x2 r8937, r8803, r8934;
}
{
add.f16x2 r8940, r8806, r8807;
}
{
mul.f16x2 r8943, r8940, r8795;
}
{
add.f16x2 r8946, r8937, r8943;
}
{
sub.f16x2 r8949, r8812, r8813;
}
{
mul.f16x2 r8952, r8949, r8794;
}
{
sub.f16x2 r8955, r8818, r8819;
}
{
mul.f16x2 r8958, r8955, r8797;
}
{
add.f16x2 r8961, r8952, r8958;
}
{
add.f16x2 %30, r8946, r8961;
}
{
add.f16x2 r8967, r8812, r8813;
}
{
mul.f16x2 r8970, r8967, r8791;
}
{
add.f16x2 r8973, r8815, r8970;
}
{
add.f16x2 r8976, r8818, r8819;
}
{
mul.f16x2 r8979, r8976, r8793;
}
{
add.f16x2 r8982, r8973, r8979;
}
{
sub.f16x2 r8985, r8800, r8801;
}
{
mul.f16x2 r8988, r8985, r8792;
}
{
sub.f16x2 r8991, r8806, r8807;
}
{
mul.f16x2 r8994, r8991, r8794;
}
{
add.f16x2 r8997, r8988, r8994;
}
{
add.f16x2 %11, r8982, r8997;
}
{
add.f16x2 r9003, r8812, r8813;
}
{
mul.f16x2 r9006, r9003, r8791;
}
{
add.f16x2 r9009, r8815, r9006;
}
{
add.f16x2 r9012, r8818, r8819;
}
{
mul.f16x2 r9015, r9012, r8793;
}
{
add.f16x2 r9018, r9009, r9015;
}
{
sub.f16x2 r9021, r8800, r8801;
}
{
mul.f16x2 r9024, r9021, r8792;
}
{
sub.f16x2 r9027, r8806, r8807;
}
{
mul.f16x2 r9030, r9027, r8794;
}
{
add.f16x2 r9033, r9024, r9030;
}
{
sub.f16x2 %41, r9018, r9033;
}
{
add.f16x2 r9039, r8812, r8813;
}
{
mul.f16x2 r9042, r9039, r8793;
}
{
add.f16x2 r9045, r8815, r9042;
}
{
add.f16x2 r9048, r8818, r8819;
}
{
mul.f16x2 r9051, r9048, r8795;
}
{
add.f16x2 r9054, r9045, r9051;
}
{
sub.f16x2 r9057, r8800, r8801;
}
{
mul.f16x2 r9060, r9057, r8794;
}
{
sub.f16x2 r9063, r8806, r8807;
}
{
mul.f16x2 r9066, r9063, r8797;
}
{
add.f16x2 r9069, r9060, r9066;
}
{
add.f16x2 %21, r9054, r9069;
}
{
add.f16x2 r9075, r8812, r8813;
}
{
mul.f16x2 r9078, r9075, r8793;
}
{
add.f16x2 r9081, r8815, r9078;
}
{
add.f16x2 r9084, r8818, r8819;
}
{
mul.f16x2 r9087, r9084, r8795;
}
{
add.f16x2 r9090, r9081, r9087;
}
{
sub.f16x2 r9093, r8800, r8801;
}
{
mul.f16x2 r9096, r9093, r8794;
}
{
sub.f16x2 r9099, r8806, r8807;
}
{
mul.f16x2 r9102, r9099, r8797;
}
{
add.f16x2 r9105, r9096, r9102;
}
{
sub.f16x2 %31, r9090, r9105;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9111, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9112, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9113, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9114, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9115, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9116, {low, high};
}
{
neg.f16x2 r9117, r9116;
}
{
add.f16x2 r9119, r9120, r9121;
}
{
add.f16x2 r9122, r9123, r9119;
}
{
add.f16x2 r9125, r9126, r9127;
}
{
add.f16x2 %2, r9122, r9125;
}
{
add.f16x2 r9131, r9132, r9133;
}
{
add.f16x2 r9134, r9135, r9131;
}
{
add.f16x2 r9137, r9138, r9139;
}
{
add.f16x2 %3, r9134, r9137;
}
{
add.f16x2 r9143, r9120, r9121;
}
{
mul.f16x2 r9146, r9143, r9111;
}
{
add.f16x2 r9149, r9123, r9146;
}
{
add.f16x2 r9152, r9126, r9127;
}
{
mul.f16x2 r9155, r9152, r9113;
}
{
add.f16x2 r9158, r9149, r9155;
}
{
sub.f16x2 r9161, r9132, r9133;
}
{
mul.f16x2 r9164, r9161, r9112;
}
{
sub.f16x2 r9167, r9138, r9139;
}
{
mul.f16x2 r9170, r9167, r9114;
}
{
add.f16x2 r9173, r9164, r9170;
}
{
sub.f16x2 %12, r9158, r9173;
}
{
add.f16x2 r9179, r9120, r9121;
}
{
mul.f16x2 r9182, r9179, r9111;
}
{
add.f16x2 r9185, r9123, r9182;
}
{
add.f16x2 r9188, r9126, r9127;
}
{
mul.f16x2 r9191, r9188, r9113;
}
{
add.f16x2 r9194, r9185, r9191;
}
{
sub.f16x2 r9197, r9132, r9133;
}
{
mul.f16x2 r9200, r9197, r9112;
}
{
sub.f16x2 r9203, r9138, r9139;
}
{
mul.f16x2 r9206, r9203, r9114;
}
{
add.f16x2 r9209, r9200, r9206;
}
{
add.f16x2 %42, r9194, r9209;
}
{
add.f16x2 r9215, r9120, r9121;
}
{
mul.f16x2 r9218, r9215, r9113;
}
{
add.f16x2 r9221, r9123, r9218;
}
{
add.f16x2 r9224, r9126, r9127;
}
{
mul.f16x2 r9227, r9224, r9115;
}
{
add.f16x2 r9230, r9221, r9227;
}
{
sub.f16x2 r9233, r9132, r9133;
}
{
mul.f16x2 r9236, r9233, r9114;
}
{
sub.f16x2 r9239, r9138, r9139;
}
{
mul.f16x2 r9242, r9239, r9117;
}
{
add.f16x2 r9245, r9236, r9242;
}
{
sub.f16x2 %22, r9230, r9245;
}
{
add.f16x2 r9251, r9120, r9121;
}
{
mul.f16x2 r9254, r9251, r9113;
}
{
add.f16x2 r9257, r9123, r9254;
}
{
add.f16x2 r9260, r9126, r9127;
}
{
mul.f16x2 r9263, r9260, r9115;
}
{
add.f16x2 r9266, r9257, r9263;
}
{
sub.f16x2 r9269, r9132, r9133;
}
{
mul.f16x2 r9272, r9269, r9114;
}
{
sub.f16x2 r9275, r9138, r9139;
}
{
mul.f16x2 r9278, r9275, r9117;
}
{
add.f16x2 r9281, r9272, r9278;
}
{
add.f16x2 %32, r9266, r9281;
}
{
add.f16x2 r9287, r9132, r9133;
}
{
mul.f16x2 r9290, r9287, r9111;
}
{
add.f16x2 r9293, r9135, r9290;
}
{
add.f16x2 r9296, r9138, r9139;
}
{
mul.f16x2 r9299, r9296, r9113;
}
{
add.f16x2 r9302, r9293, r9299;
}
{
sub.f16x2 r9305, r9120, r9121;
}
{
mul.f16x2 r9308, r9305, r9112;
}
{
sub.f16x2 r9311, r9126, r9127;
}
{
mul.f16x2 r9314, r9311, r9114;
}
{
add.f16x2 r9317, r9308, r9314;
}
{
add.f16x2 %13, r9302, r9317;
}
{
add.f16x2 r9323, r9132, r9133;
}
{
mul.f16x2 r9326, r9323, r9111;
}
{
add.f16x2 r9329, r9135, r9326;
}
{
add.f16x2 r9332, r9138, r9139;
}
{
mul.f16x2 r9335, r9332, r9113;
}
{
add.f16x2 r9338, r9329, r9335;
}
{
sub.f16x2 r9341, r9120, r9121;
}
{
mul.f16x2 r9344, r9341, r9112;
}
{
sub.f16x2 r9347, r9126, r9127;
}
{
mul.f16x2 r9350, r9347, r9114;
}
{
add.f16x2 r9353, r9344, r9350;
}
{
sub.f16x2 %43, r9338, r9353;
}
{
add.f16x2 r9359, r9132, r9133;
}
{
mul.f16x2 r9362, r9359, r9113;
}
{
add.f16x2 r9365, r9135, r9362;
}
{
add.f16x2 r9368, r9138, r9139;
}
{
mul.f16x2 r9371, r9368, r9115;
}
{
add.f16x2 r9374, r9365, r9371;
}
{
sub.f16x2 r9377, r9120, r9121;
}
{
mul.f16x2 r9380, r9377, r9114;
}
{
sub.f16x2 r9383, r9126, r9127;
}
{
mul.f16x2 r9386, r9383, r9117;
}
{
add.f16x2 r9389, r9380, r9386;
}
{
add.f16x2 %23, r9374, r9389;
}
{
add.f16x2 r9395, r9132, r9133;
}
{
mul.f16x2 r9398, r9395, r9113;
}
{
add.f16x2 r9401, r9135, r9398;
}
{
add.f16x2 r9404, r9138, r9139;
}
{
mul.f16x2 r9407, r9404, r9115;
}
{
add.f16x2 r9410, r9401, r9407;
}
{
sub.f16x2 r9413, r9120, r9121;
}
{
mul.f16x2 r9416, r9413, r9114;
}
{
sub.f16x2 r9419, r9126, r9127;
}
{
mul.f16x2 r9422, r9419, r9117;
}
{
add.f16x2 r9425, r9416, r9422;
}
{
sub.f16x2 %33, r9410, r9425;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9431, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9432, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9433, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9434, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9435, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9436, {low, high};
}
{
neg.f16x2 r9437, r9436;
}
{
add.f16x2 r9439, r9440, r9441;
}
{
add.f16x2 r9442, r9443, r9439;
}
{
add.f16x2 r9445, r9446, r9447;
}
{
add.f16x2 %4, r9442, r9445;
}
{
add.f16x2 r9451, r9452, r9453;
}
{
add.f16x2 r9454, r9455, r9451;
}
{
add.f16x2 r9457, r9458, r9459;
}
{
add.f16x2 %5, r9454, r9457;
}
{
add.f16x2 r9463, r9440, r9441;
}
{
mul.f16x2 r9466, r9463, r9431;
}
{
add.f16x2 r9469, r9443, r9466;
}
{
add.f16x2 r9472, r9446, r9447;
}
{
mul.f16x2 r9475, r9472, r9433;
}
{
add.f16x2 r9478, r9469, r9475;
}
{
sub.f16x2 r9481, r9452, r9453;
}
{
mul.f16x2 r9484, r9481, r9432;
}
{
sub.f16x2 r9487, r9458, r9459;
}
{
mul.f16x2 r9490, r9487, r9434;
}
{
add.f16x2 r9493, r9484, r9490;
}
{
sub.f16x2 %14, r9478, r9493;
}
{
add.f16x2 r9499, r9440, r9441;
}
{
mul.f16x2 r9502, r9499, r9431;
}
{
add.f16x2 r9505, r9443, r9502;
}
{
add.f16x2 r9508, r9446, r9447;
}
{
mul.f16x2 r9511, r9508, r9433;
}
{
add.f16x2 r9514, r9505, r9511;
}
{
sub.f16x2 r9517, r9452, r9453;
}
{
mul.f16x2 r9520, r9517, r9432;
}
{
sub.f16x2 r9523, r9458, r9459;
}
{
mul.f16x2 r9526, r9523, r9434;
}
{
add.f16x2 r9529, r9520, r9526;
}
{
add.f16x2 %44, r9514, r9529;
}
{
add.f16x2 r9535, r9440, r9441;
}
{
mul.f16x2 r9538, r9535, r9433;
}
{
add.f16x2 r9541, r9443, r9538;
}
{
add.f16x2 r9544, r9446, r9447;
}
{
mul.f16x2 r9547, r9544, r9435;
}
{
add.f16x2 r9550, r9541, r9547;
}
{
sub.f16x2 r9553, r9452, r9453;
}
{
mul.f16x2 r9556, r9553, r9434;
}
{
sub.f16x2 r9559, r9458, r9459;
}
{
mul.f16x2 r9562, r9559, r9437;
}
{
add.f16x2 r9565, r9556, r9562;
}
{
sub.f16x2 %24, r9550, r9565;
}
{
add.f16x2 r9571, r9440, r9441;
}
{
mul.f16x2 r9574, r9571, r9433;
}
{
add.f16x2 r9577, r9443, r9574;
}
{
add.f16x2 r9580, r9446, r9447;
}
{
mul.f16x2 r9583, r9580, r9435;
}
{
add.f16x2 r9586, r9577, r9583;
}
{
sub.f16x2 r9589, r9452, r9453;
}
{
mul.f16x2 r9592, r9589, r9434;
}
{
sub.f16x2 r9595, r9458, r9459;
}
{
mul.f16x2 r9598, r9595, r9437;
}
{
add.f16x2 r9601, r9592, r9598;
}
{
add.f16x2 %34, r9586, r9601;
}
{
add.f16x2 r9607, r9452, r9453;
}
{
mul.f16x2 r9610, r9607, r9431;
}
{
add.f16x2 r9613, r9455, r9610;
}
{
add.f16x2 r9616, r9458, r9459;
}
{
mul.f16x2 r9619, r9616, r9433;
}
{
add.f16x2 r9622, r9613, r9619;
}
{
sub.f16x2 r9625, r9440, r9441;
}
{
mul.f16x2 r9628, r9625, r9432;
}
{
sub.f16x2 r9631, r9446, r9447;
}
{
mul.f16x2 r9634, r9631, r9434;
}
{
add.f16x2 r9637, r9628, r9634;
}
{
add.f16x2 %15, r9622, r9637;
}
{
add.f16x2 r9643, r9452, r9453;
}
{
mul.f16x2 r9646, r9643, r9431;
}
{
add.f16x2 r9649, r9455, r9646;
}
{
add.f16x2 r9652, r9458, r9459;
}
{
mul.f16x2 r9655, r9652, r9433;
}
{
add.f16x2 r9658, r9649, r9655;
}
{
sub.f16x2 r9661, r9440, r9441;
}
{
mul.f16x2 r9664, r9661, r9432;
}
{
sub.f16x2 r9667, r9446, r9447;
}
{
mul.f16x2 r9670, r9667, r9434;
}
{
add.f16x2 r9673, r9664, r9670;
}
{
sub.f16x2 %45, r9658, r9673;
}
{
add.f16x2 r9679, r9452, r9453;
}
{
mul.f16x2 r9682, r9679, r9433;
}
{
add.f16x2 r9685, r9455, r9682;
}
{
add.f16x2 r9688, r9458, r9459;
}
{
mul.f16x2 r9691, r9688, r9435;
}
{
add.f16x2 r9694, r9685, r9691;
}
{
sub.f16x2 r9697, r9440, r9441;
}
{
mul.f16x2 r9700, r9697, r9434;
}
{
sub.f16x2 r9703, r9446, r9447;
}
{
mul.f16x2 r9706, r9703, r9437;
}
{
add.f16x2 r9709, r9700, r9706;
}
{
add.f16x2 %25, r9694, r9709;
}
{
add.f16x2 r9715, r9452, r9453;
}
{
mul.f16x2 r9718, r9715, r9433;
}
{
add.f16x2 r9721, r9455, r9718;
}
{
add.f16x2 r9724, r9458, r9459;
}
{
mul.f16x2 r9727, r9724, r9435;
}
{
add.f16x2 r9730, r9721, r9727;
}
{
sub.f16x2 r9733, r9440, r9441;
}
{
mul.f16x2 r9736, r9733, r9434;
}
{
sub.f16x2 r9739, r9446, r9447;
}
{
mul.f16x2 r9742, r9739, r9437;
}
{
add.f16x2 r9745, r9736, r9742;
}
{
sub.f16x2 %35, r9730, r9745;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9751, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9752, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r9753, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r9754, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r9755, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r9756, {low, high};
}
{
neg.f16x2 r9757, r9756;
}
{
add.f16x2 r9759, r9760, r9761;
}
{
add.f16x2 r9762, r9763, r9759;
}
{
add.f16x2 r9765, r9766, r9767;
}
{
add.f16x2 %6, r9762, r9765;
}
{
add.f16x2 r9771, r9772, r9773;
}
{
add.f16x2 r9774, r9775, r9771;
}
{
add.f16x2 r9777, r9778, r9779;
}
{
add.f16x2 %7, r9774, r9777;
}
{
add.f16x2 r9783, r9760, r9761;
}
{
mul.f16x2 r9786, r9783, r9751;
}
{
add.f16x2 r9789, r9763, r9786;
}
{
add.f16x2 r9792, r9766, r9767;
}
{
mul.f16x2 r9795, r9792, r9753;
}
{
add.f16x2 r9798, r9789, r9795;
}
{
sub.f16x2 r9801, r9772, r9773;
}
{
mul.f16x2 r9804, r9801, r9752;
}
{
sub.f16x2 r9807, r9778, r9779;
}
{
mul.f16x2 r9810, r9807, r9754;
}
{
add.f16x2 r9813, r9804, r9810;
}
{
sub.f16x2 %16, r9798, r9813;
}
{
add.f16x2 r9819, r9760, r9761;
}
{
mul.f16x2 r9822, r9819, r9751;
}
{
add.f16x2 r9825, r9763, r9822;
}
{
add.f16x2 r9828, r9766, r9767;
}
{
mul.f16x2 r9831, r9828, r9753;
}
{
add.f16x2 r9834, r9825, r9831;
}
{
sub.f16x2 r9837, r9772, r9773;
}
{
mul.f16x2 r9840, r9837, r9752;
}
{
sub.f16x2 r9843, r9778, r9779;
}
{
mul.f16x2 r9846, r9843, r9754;
}
{
add.f16x2 r9849, r9840, r9846;
}
{
add.f16x2 %46, r9834, r9849;
}
{
add.f16x2 r9855, r9760, r9761;
}
{
mul.f16x2 r9858, r9855, r9753;
}
{
add.f16x2 r9861, r9763, r9858;
}
{
add.f16x2 r9864, r9766, r9767;
}
{
mul.f16x2 r9867, r9864, r9755;
}
{
add.f16x2 r9870, r9861, r9867;
}
{
sub.f16x2 r9873, r9772, r9773;
}
{
mul.f16x2 r9876, r9873, r9754;
}
{
sub.f16x2 r9879, r9778, r9779;
}
{
mul.f16x2 r9882, r9879, r9757;
}
{
add.f16x2 r9885, r9876, r9882;
}
{
sub.f16x2 %26, r9870, r9885;
}
{
add.f16x2 r9891, r9760, r9761;
}
{
mul.f16x2 r9894, r9891, r9753;
}
{
add.f16x2 r9897, r9763, r9894;
}
{
add.f16x2 r9900, r9766, r9767;
}
{
mul.f16x2 r9903, r9900, r9755;
}
{
add.f16x2 r9906, r9897, r9903;
}
{
sub.f16x2 r9909, r9772, r9773;
}
{
mul.f16x2 r9912, r9909, r9754;
}
{
sub.f16x2 r9915, r9778, r9779;
}
{
mul.f16x2 r9918, r9915, r9757;
}
{
add.f16x2 r9921, r9912, r9918;
}
{
add.f16x2 %36, r9906, r9921;
}
{
add.f16x2 r9927, r9772, r9773;
}
{
mul.f16x2 r9930, r9927, r9751;
}
{
add.f16x2 r9933, r9775, r9930;
}
{
add.f16x2 r9936, r9778, r9779;
}
{
mul.f16x2 r9939, r9936, r9753;
}
{
add.f16x2 r9942, r9933, r9939;
}
{
sub.f16x2 r9945, r9760, r9761;
}
{
mul.f16x2 r9948, r9945, r9752;
}
{
sub.f16x2 r9951, r9766, r9767;
}
{
mul.f16x2 r9954, r9951, r9754;
}
{
add.f16x2 r9957, r9948, r9954;
}
{
add.f16x2 %17, r9942, r9957;
}
{
add.f16x2 r9963, r9772, r9773;
}
{
mul.f16x2 r9966, r9963, r9751;
}
{
add.f16x2 r9969, r9775, r9966;
}
{
add.f16x2 r9972, r9778, r9779;
}
{
mul.f16x2 r9975, r9972, r9753;
}
{
add.f16x2 r9978, r9969, r9975;
}
{
sub.f16x2 r9981, r9760, r9761;
}
{
mul.f16x2 r9984, r9981, r9752;
}
{
sub.f16x2 r9987, r9766, r9767;
}
{
mul.f16x2 r9990, r9987, r9754;
}
{
add.f16x2 r9993, r9984, r9990;
}
{
sub.f16x2 %47, r9978, r9993;
}
{
add.f16x2 r9999, r9772, r9773;
}
{
mul.f16x2 r10002, r9999, r9753;
}
{
add.f16x2 r10005, r9775, r10002;
}
{
add.f16x2 r10008, r9778, r9779;
}
{
mul.f16x2 r10011, r10008, r9755;
}
{
add.f16x2 r10014, r10005, r10011;
}
{
sub.f16x2 r10017, r9760, r9761;
}
{
mul.f16x2 r10020, r10017, r9754;
}
{
sub.f16x2 r10023, r9766, r9767;
}
{
mul.f16x2 r10026, r10023, r9757;
}
{
add.f16x2 r10029, r10020, r10026;
}
{
add.f16x2 %27, r10014, r10029;
}
{
add.f16x2 r10035, r9772, r9773;
}
{
mul.f16x2 r10038, r10035, r9753;
}
{
add.f16x2 r10041, r9775, r10038;
}
{
add.f16x2 r10044, r9778, r9779;
}
{
mul.f16x2 r10047, r10044, r9755;
}
{
add.f16x2 r10050, r10041, r10047;
}
{
sub.f16x2 r10053, r9760, r9761;
}
{
mul.f16x2 r10056, r10053, r9754;
}
{
sub.f16x2 r10059, r9766, r9767;
}
{
mul.f16x2 r10062, r10059, r9757;
}
{
add.f16x2 r10065, r10056, r10062;
}
{
sub.f16x2 %37, r10050, r10065;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10071, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10072, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f590;
cvt.rn.f16.f32 high, f590;
mov.b32 r10073, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f592;
cvt.rn.f16.f32 high, f592;
mov.b32 r10074, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f594;
cvt.rn.f16.f32 high, f594;
mov.b32 r10075, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f596;
cvt.rn.f16.f32 high, f596;
mov.b32 r10076, {low, high};
}
{
neg.f16x2 r10077, r10076;
}
{
add.f16x2 r10079, r10080, r10081;
}
{
add.f16x2 r10082, r10083, r10079;
}
{
add.f16x2 r10085, r10086, r10087;
}
{
add.f16x2 %8, r10082, r10085;
}
{
add.f16x2 r10091, r10092, r10093;
}
{
add.f16x2 r10094, r10095, r10091;
}
{
add.f16x2 r10097, r10098, r10099;
}
{
add.f16x2 %9, r10094, r10097;
}
{
add.f16x2 r10103, r10080, r10081;
}
{
mul.f16x2 r10106, r10103, r10071;
}
{
add.f16x2 r10109, r10083, r10106;
}
{
add.f16x2 r10112, r10086, r10087;
}
{
mul.f16x2 r10115, r10112, r10073;
}
{
add.f16x2 r10118, r10109, r10115;
}
{
sub.f16x2 r10121, r10092, r10093;
}
{
mul.f16x2 r10124, r10121, r10072;
}
{
sub.f16x2 r10127, r10098, r10099;
}
{
mul.f16x2 r10130, r10127, r10074;
}
{
add.f16x2 r10133, r10124, r10130;
}
{
sub.f16x2 %18, r10118, r10133;
}
{
add.f16x2 r10139, r10080, r10081;
}
{
mul.f16x2 r10142, r10139, r10071;
}
{
add.f16x2 r10145, r10083, r10142;
}
{
add.f16x2 r10148, r10086, r10087;
}
{
mul.f16x2 r10151, r10148, r10073;
}
{
add.f16x2 r10154, r10145, r10151;
}
{
sub.f16x2 r10157, r10092, r10093;
}
{
mul.f16x2 r10160, r10157, r10072;
}
{
sub.f16x2 r10163, r10098, r10099;
}
{
mul.f16x2 r10166, r10163, r10074;
}
{
add.f16x2 r10169, r10160, r10166;
}
{
add.f16x2 %48, r10154, r10169;
}
{
add.f16x2 r10175, r10080, r10081;
}
{
mul.f16x2 r10178, r10175, r10073;
}
{
add.f16x2 r10181, r10083, r10178;
}
{
add.f16x2 r10184, r10086, r10087;
}
{
mul.f16x2 r10187, r10184, r10075;
}
{
add.f16x2 r10190, r10181, r10187;
}
{
sub.f16x2 r10193, r10092, r10093;
}
{
mul.f16x2 r10196, r10193, r10074;
}
{
sub.f16x2 r10199, r10098, r10099;
}
{
mul.f16x2 r10202, r10199, r10077;
}
{
add.f16x2 r10205, r10196, r10202;
}
{
sub.f16x2 %28, r10190, r10205;
}
{
add.f16x2 r10211, r10080, r10081;
}
{
mul.f16x2 r10214, r10211, r10073;
}
{
add.f16x2 r10217, r10083, r10214;
}
{
add.f16x2 r10220, r10086, r10087;
}
{
mul.f16x2 r10223, r10220, r10075;
}
{
add.f16x2 r10226, r10217, r10223;
}
{
sub.f16x2 r10229, r10092, r10093;
}
{
mul.f16x2 r10232, r10229, r10074;
}
{
sub.f16x2 r10235, r10098, r10099;
}
{
mul.f16x2 r10238, r10235, r10077;
}
{
add.f16x2 r10241, r10232, r10238;
}
{
add.f16x2 %38, r10226, r10241;
}
{
add.f16x2 r10247, r10092, r10093;
}
{
mul.f16x2 r10250, r10247, r10071;
}
{
add.f16x2 r10253, r10095, r10250;
}
{
add.f16x2 r10256, r10098, r10099;
}
{
mul.f16x2 r10259, r10256, r10073;
}
{
add.f16x2 r10262, r10253, r10259;
}
{
sub.f16x2 r10265, r10080, r10081;
}
{
mul.f16x2 r10268, r10265, r10072;
}
{
sub.f16x2 r10271, r10086, r10087;
}
{
mul.f16x2 r10274, r10271, r10074;
}
{
add.f16x2 r10277, r10268, r10274;
}
{
add.f16x2 %19, r10262, r10277;
}
{
add.f16x2 r10283, r10092, r10093;
}
{
mul.f16x2 r10286, r10283, r10071;
}
{
add.f16x2 r10289, r10095, r10286;
}
{
add.f16x2 r10292, r10098, r10099;
}
{
mul.f16x2 r10295, r10292, r10073;
}
{
add.f16x2 r10298, r10289, r10295;
}
{
sub.f16x2 r10301, r10080, r10081;
}
{
mul.f16x2 r10304, r10301, r10072;
}
{
sub.f16x2 r10307, r10086, r10087;
}
{
mul.f16x2 r10310, r10307, r10074;
}
{
add.f16x2 r10313, r10304, r10310;
}
{
sub.f16x2 %49, r10298, r10313;
}
{
add.f16x2 r10319, r10092, r10093;
}
{
mul.f16x2 r10322, r10319, r10073;
}
{
add.f16x2 r10325, r10095, r10322;
}
{
add.f16x2 r10328, r10098, r10099;
}
{
mul.f16x2 r10331, r10328, r10075;
}
{
add.f16x2 r10334, r10325, r10331;
}
{
sub.f16x2 r10337, r10080, r10081;
}
{
mul.f16x2 r10340, r10337, r10074;
}
{
sub.f16x2 r10343, r10086, r10087;
}
{
mul.f16x2 r10346, r10343, r10077;
}
{
add.f16x2 r10349, r10340, r10346;
}
{
add.f16x2 %29, r10334, r10349;
}
{
add.f16x2 r10355, r10092, r10093;
}
{
mul.f16x2 r10358, r10355, r10073;
}
{
add.f16x2 r10361, r10095, r10358;
}
{
add.f16x2 r10364, r10098, r10099;
}
{
mul.f16x2 r10367, r10364, r10075;
}
{
add.f16x2 r10370, r10361, r10367;
}
{
sub.f16x2 r10373, r10080, r10081;
}
{
mul.f16x2 r10376, r10373, r10074;
}
{
sub.f16x2 r10379, r10086, r10087;
}
{
mul.f16x2 r10382, r10379, r10077;
}
{
add.f16x2 r10385, r10376, r10382;
}
{
sub.f16x2 %39, r10370, r10385;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[18].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<914, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<121>;
.reg .b32 r<2234>;
.reg .b64 rd<10>;
mov.u32 r2205, %tid.x;
mov.f32 f106, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1, {low, high};
}
mov.f32 f108, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r2, {low, high};
}
mov.f32 f102, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r3, {low, high};
}
mov.f32 f104, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %13, %19;
}
{
add.f16x2 r12, %11, r9;
}
{
add.f16x2 r15, %15, %17;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %14, %20;
}
{
add.f16x2 r24, %12, r21;
}
{
add.f16x2 r27, %16, %18;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %13, %19;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %11, r36;
}
{
add.f16x2 r42, %15, %17;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %14, %20;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %16, %18;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %13, %19;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %11, r72;
}
{
add.f16x2 r78, %15, %17;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %14, %20;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %16, %18;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %13, %19;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %11, r108;
}
{
add.f16x2 r114, %15, %17;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %14, %20;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %16, %18;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %13, %19;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %11, r144;
}
{
add.f16x2 r150, %15, %17;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %14, %20;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %16, %18;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %14, %20;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %12, r180;
}
{
add.f16x2 r186, %16, %18;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %13, %19;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %15, %17;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %14, %20;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %12, r216;
}
{
add.f16x2 r222, %16, %18;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %13, %19;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %15, %17;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %14, %20;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %12, r252;
}
{
add.f16x2 r258, %16, %18;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %13, %19;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %15, %17;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %14, %20;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %12, r288;
}
{
add.f16x2 r294, %16, %18;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %13, %19;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %15, %17;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
mul.wide.u32 rd2, r2205, -776530087;
shr.u64 rd3, rd2, 41;
cvt.u32.u64 r2206, rd3;
mul.lo.s32 r2207, r2206, 625;
sub.s32 r2208, r2205, r2207;
cvt.rn.f32.u32 f109, r2208;
mul.f32 f110, f109, 0f3B03C498;
cos.approx.f32 f13, f110;
sin.approx.f32 f111, f110;
neg.f32 f14, f111;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r321, {low, high};
}
mov.u32 r2209, %tid.y;
mov.u32 r2210, %10;
mad.lo.s32 r2211, r2209, 25000, r2210;
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r324, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r326, {high, high};
}
{
mul.f16x2 r328, r210, r326;
}
{
neg.f16x2 r331, r328;
}
{
fma.rn.f16x2 r333, r66, r324, r331;
}
{
mul.f16x2 r337, r66, r326;
}
{
fma.rn.f16x2 r340, r210, r324, r337;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r344, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r346, {high, high};
}
mov.f32 f93, 0fBF800000;
mov.f32 f94, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r348, {low, high};
}
{
mul.f16x2 r349, r346, r348;
}
{
mul.f16x2 r352, r321, r344;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r355, {high, low};
}
{
fma.rn.f16x2 r357, r349, r355, r352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r361, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r363, {high, high};
}
{
mul.f16x2 r365, r282, r363;
}
{
neg.f16x2 r368, r365;
}
{
fma.rn.f16x2 r370, r138, r361, r368;
}
{
mul.f16x2 r374, r138, r363;
}
{
fma.rn.f16x2 r377, r282, r361, r374;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r381, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r383, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r385, {low, high};
}
{
mul.f16x2 r386, r383, r385;
}
{
mul.f16x2 r389, r357, r381;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r392, {high, low};
}
{
fma.rn.f16x2 r394, r386, r392, r389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r398, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r400, {high, high};
}
{
mul.f16x2 r402, r318, r400;
}
{
neg.f16x2 r405, r402;
}
{
fma.rn.f16x2 r407, r174, r398, r405;
}
{
mul.f16x2 r411, r174, r400;
}
{
fma.rn.f16x2 r414, r318, r398, r411;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r418, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r420, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r422, {low, high};
}
{
mul.f16x2 r423, r420, r422;
}
{
mul.f16x2 r426, r394, r418;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r429, {high, low};
}
{
fma.rn.f16x2 r431, r423, r429, r426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r431;
mov.b32 r435, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r431;
mov.b32 r437, {high, high};
}
{
mul.f16x2 r439, r246, r437;
}
{
neg.f16x2 r442, r439;
}
{
fma.rn.f16x2 r444, r102, r435, r442;
}
{
mul.f16x2 r448, r102, r437;
}
{
fma.rn.f16x2 r451, r246, r435, r448;
}
mad.lo.s32 r2212, r2206, 25000, r2211;
barrier.sync 0;
mad.lo.s32 r2213, r2208, 40, r2212;
st.shared.v2.f32 [r2213], {r18, r30};
st.shared.v2.f32 [r2213+8], {r333, r340};
st.shared.v2.f32 [r2213+16], {r370, r377};
st.shared.v2.f32 [r2213+24], {r407, r414};
st.shared.v2.f32 [r2213+32], {r444, r451};
barrier.sync 0;
shl.b32 r2214, r2208, 5;
sub.s32 r2215, r2213, r2214;
ld.shared.u32 r484, [r2215];
ld.shared.u32 r496, [r2215+4];
ld.shared.u32 r481, [r2215+5000];
ld.shared.u32 r493, [r2215+5004];
ld.shared.u32 r487, [r2215+10000];
ld.shared.u32 r499, [r2215+10004];
ld.shared.u32 r488, [r2215+15000];
ld.shared.u32 r500, [r2215+15004];
ld.shared.u32 r482, [r2215+20000];
ld.shared.u32 r494, [r2215+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r477, {low, high};
}
{
neg.f16x2 r478, r477;
}
{
add.f16x2 r480, r481, r482;
}
{
add.f16x2 r483, r484, r480;
}
{
add.f16x2 r486, r487, r488;
}
{
add.f16x2 r489, r483, r486;
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 r495, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 r501, r495, r498;
}
{
add.f16x2 r504, r481, r482;
}
{
mul.f16x2 r507, r504, r472;
}
{
add.f16x2 r510, r484, r507;
}
{
add.f16x2 r513, r487, r488;
}
{
mul.f16x2 r516, r513, r474;
}
{
add.f16x2 r519, r510, r516;
}
{
sub.f16x2 r522, r493, r494;
}
{
mul.f16x2 r525, r522, r473;
}
{
sub.f16x2 r528, r499, r500;
}
{
mul.f16x2 r531, r528, r475;
}
{
add.f16x2 r534, r525, r531;
}
{
sub.f16x2 r537, r519, r534;
}
{
add.f16x2 r540, r481, r482;
}
{
mul.f16x2 r543, r540, r472;
}
{
add.f16x2 r546, r484, r543;
}
{
add.f16x2 r549, r487, r488;
}
{
mul.f16x2 r552, r549, r474;
}
{
add.f16x2 r555, r546, r552;
}
{
sub.f16x2 r558, r493, r494;
}
{
mul.f16x2 r561, r558, r473;
}
{
sub.f16x2 r564, r499, r500;
}
{
mul.f16x2 r567, r564, r475;
}
{
add.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r555, r570;
}
{
add.f16x2 r576, r481, r482;
}
{
mul.f16x2 r579, r576, r474;
}
{
add.f16x2 r582, r484, r579;
}
{
add.f16x2 r585, r487, r488;
}
{
mul.f16x2 r588, r585, r476;
}
{
add.f16x2 r591, r582, r588;
}
{
sub.f16x2 r594, r493, r494;
}
{
mul.f16x2 r597, r594, r475;
}
{
sub.f16x2 r600, r499, r500;
}
{
mul.f16x2 r603, r600, r478;
}
{
add.f16x2 r606, r597, r603;
}
{
sub.f16x2 r609, r591, r606;
}
{
add.f16x2 r612, r481, r482;
}
{
mul.f16x2 r615, r612, r474;
}
{
add.f16x2 r618, r484, r615;
}
{
add.f16x2 r621, r487, r488;
}
{
mul.f16x2 r624, r621, r476;
}
{
add.f16x2 r627, r618, r624;
}
{
sub.f16x2 r630, r493, r494;
}
{
mul.f16x2 r633, r630, r475;
}
{
sub.f16x2 r636, r499, r500;
}
{
mul.f16x2 r639, r636, r478;
}
{
add.f16x2 r642, r633, r639;
}
{
add.f16x2 r645, r627, r642;
}
{
add.f16x2 r648, r493, r494;
}
{
mul.f16x2 r651, r648, r472;
}
{
add.f16x2 r654, r496, r651;
}
{
add.f16x2 r657, r499, r500;
}
{
mul.f16x2 r660, r657, r474;
}
{
add.f16x2 r663, r654, r660;
}
{
sub.f16x2 r666, r481, r482;
}
{
mul.f16x2 r669, r666, r473;
}
{
sub.f16x2 r672, r487, r488;
}
{
mul.f16x2 r675, r672, r475;
}
{
add.f16x2 r678, r669, r675;
}
{
add.f16x2 r681, r663, r678;
}
{
add.f16x2 r684, r493, r494;
}
{
mul.f16x2 r687, r684, r472;
}
{
add.f16x2 r690, r496, r687;
}
{
add.f16x2 r693, r499, r500;
}
{
mul.f16x2 r696, r693, r474;
}
{
add.f16x2 r699, r690, r696;
}
{
sub.f16x2 r702, r481, r482;
}
{
mul.f16x2 r705, r702, r473;
}
{
sub.f16x2 r708, r487, r488;
}
{
mul.f16x2 r711, r708, r475;
}
{
add.f16x2 r714, r705, r711;
}
{
sub.f16x2 r717, r699, r714;
}
{
add.f16x2 r720, r493, r494;
}
{
mul.f16x2 r723, r720, r474;
}
{
add.f16x2 r726, r496, r723;
}
{
add.f16x2 r729, r499, r500;
}
{
mul.f16x2 r732, r729, r476;
}
{
add.f16x2 r735, r726, r732;
}
{
sub.f16x2 r738, r481, r482;
}
{
mul.f16x2 r741, r738, r475;
}
{
sub.f16x2 r744, r487, r488;
}
{
mul.f16x2 r747, r744, r478;
}
{
add.f16x2 r750, r741, r747;
}
{
add.f16x2 r753, r735, r750;
}
{
add.f16x2 r756, r493, r494;
}
{
mul.f16x2 r759, r756, r474;
}
{
add.f16x2 r762, r496, r759;
}
{
add.f16x2 r765, r499, r500;
}
{
mul.f16x2 r768, r765, r476;
}
{
add.f16x2 r771, r762, r768;
}
{
sub.f16x2 r774, r481, r482;
}
{
mul.f16x2 r777, r774, r475;
}
{
sub.f16x2 r780, r487, r488;
}
{
mul.f16x2 r783, r780, r478;
}
{
add.f16x2 r786, r777, r783;
}
{
sub.f16x2 r789, r771, r786;
}
mul.wide.u32 rd4, r2208, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r2216, rd5;
cvt.rn.f32.u32 f112, r2216;
mul.f32 f113, f112, 0f3C24B5BE;
cos.approx.f32 f37, f113;
sin.approx.f32 f114, f113;
neg.f32 f38, f114;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r792, {low, high};
}
mul.lo.s32 r2217, r2216, 5;
sub.s32 r2218, r2208, r2217;
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r795, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r797, {high, high};
}
{
mul.f16x2 r799, r681, r797;
}
{
neg.f16x2 r802, r799;
}
{
fma.rn.f16x2 r804, r537, r795, r802;
}
{
mul.f16x2 r808, r537, r797;
}
{
fma.rn.f16x2 r811, r681, r795, r808;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r815, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r817, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r819, {low, high};
}
{
mul.f16x2 r820, r817, r819;
}
{
mul.f16x2 r823, r792, r815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r826, {high, low};
}
{
fma.rn.f16x2 r828, r820, r826, r823;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r832, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r834, {high, high};
}
{
mul.f16x2 r836, r753, r834;
}
{
neg.f16x2 r839, r836;
}
{
fma.rn.f16x2 r841, r609, r832, r839;
}
{
mul.f16x2 r845, r609, r834;
}
{
fma.rn.f16x2 r848, r753, r832, r845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r852, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r854, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r856, {low, high};
}
{
mul.f16x2 r857, r854, r856;
}
{
mul.f16x2 r860, r828, r852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r863, {high, low};
}
{
fma.rn.f16x2 r865, r857, r863, r860;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r869, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r871, {high, high};
}
{
mul.f16x2 r873, r789, r871;
}
{
neg.f16x2 r876, r873;
}
{
fma.rn.f16x2 r878, r645, r869, r876;
}
{
mul.f16x2 r882, r645, r871;
}
{
fma.rn.f16x2 r885, r789, r869, r882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r889, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r891, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r893, {low, high};
}
{
mul.f16x2 r894, r891, r893;
}
{
mul.f16x2 r897, r865, r889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r900, {high, low};
}
{
fma.rn.f16x2 r902, r894, r900, r897;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r906, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r908, {high, high};
}
{
mul.f16x2 r910, r717, r908;
}
{
neg.f16x2 r913, r910;
}
{
fma.rn.f16x2 r915, r573, r906, r913;
}
{
mul.f16x2 r919, r573, r908;
}
{
fma.rn.f16x2 r922, r717, r906, r919;
}
shl.b32 r2219, r2218, 3;
add.s32 r2220, r2212, r2219;
barrier.sync 0;
mad.lo.s32 r2221, r2216, 200, r2220;
st.shared.u32 [r2221], r489;
st.shared.u32 [r2221+4], r501;
st.shared.u32 [r2221+40], r804;
st.shared.u32 [r2221+44], r811;
st.shared.u32 [r2221+80], r841;
st.shared.u32 [r2221+84], r848;
st.shared.u32 [r2221+120], r878;
st.shared.u32 [r2221+124], r885;
st.shared.u32 [r2221+160], r915;
st.shared.u32 [r2221+164], r922;
barrier.sync 0;
ld.shared.u32 r955, [r2215];
ld.shared.u32 r967, [r2215+4];
ld.shared.u32 r952, [r2215+5000];
ld.shared.u32 r964, [r2215+5004];
ld.shared.u32 r958, [r2215+10000];
ld.shared.u32 r970, [r2215+10004];
ld.shared.u32 r959, [r2215+15000];
ld.shared.u32 r971, [r2215+15004];
ld.shared.u32 r953, [r2215+20000];
ld.shared.u32 r965, [r2215+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r946, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
add.f16x2 r951, r952, r953;
}
{
add.f16x2 r954, r955, r951;
}
{
add.f16x2 r957, r958, r959;
}
{
add.f16x2 r960, r954, r957;
}
{
add.f16x2 r963, r964, r965;
}
{
add.f16x2 r966, r967, r963;
}
{
add.f16x2 r969, r970, r971;
}
{
add.f16x2 r972, r966, r969;
}
{
add.f16x2 r975, r952, r953;
}
{
mul.f16x2 r978, r975, r943;
}
{
add.f16x2 r981, r955, r978;
}
{
add.f16x2 r984, r958, r959;
}
{
mul.f16x2 r987, r984, r945;
}
{
add.f16x2 r990, r981, r987;
}
{
sub.f16x2 r993, r964, r965;
}
{
mul.f16x2 r996, r993, r944;
}
{
sub.f16x2 r999, r970, r971;
}
{
mul.f16x2 r1002, r999, r946;
}
{
add.f16x2 r1005, r996, r1002;
}
{
sub.f16x2 r1008, r990, r1005;
}
{
add.f16x2 r1011, r952, r953;
}
{
mul.f16x2 r1014, r1011, r943;
}
{
add.f16x2 r1017, r955, r1014;
}
{
add.f16x2 r1020, r958, r959;
}
{
mul.f16x2 r1023, r1020, r945;
}
{
add.f16x2 r1026, r1017, r1023;
}
{
sub.f16x2 r1029, r964, r965;
}
{
mul.f16x2 r1032, r1029, r944;
}
{
sub.f16x2 r1035, r970, r971;
}
{
mul.f16x2 r1038, r1035, r946;
}
{
add.f16x2 r1041, r1032, r1038;
}
{
add.f16x2 r1044, r1026, r1041;
}
{
add.f16x2 r1047, r952, r953;
}
{
mul.f16x2 r1050, r1047, r945;
}
{
add.f16x2 r1053, r955, r1050;
}
{
add.f16x2 r1056, r958, r959;
}
{
mul.f16x2 r1059, r1056, r947;
}
{
add.f16x2 r1062, r1053, r1059;
}
{
sub.f16x2 r1065, r964, r965;
}
{
mul.f16x2 r1068, r1065, r946;
}
{
sub.f16x2 r1071, r970, r971;
}
{
mul.f16x2 r1074, r1071, r949;
}
{
add.f16x2 r1077, r1068, r1074;
}
{
sub.f16x2 r1080, r1062, r1077;
}
{
add.f16x2 r1083, r952, r953;
}
{
mul.f16x2 r1086, r1083, r945;
}
{
add.f16x2 r1089, r955, r1086;
}
{
add.f16x2 r1092, r958, r959;
}
{
mul.f16x2 r1095, r1092, r947;
}
{
add.f16x2 r1098, r1089, r1095;
}
{
sub.f16x2 r1101, r964, r965;
}
{
mul.f16x2 r1104, r1101, r946;
}
{
sub.f16x2 r1107, r970, r971;
}
{
mul.f16x2 r1110, r1107, r949;
}
{
add.f16x2 r1113, r1104, r1110;
}
{
add.f16x2 r1116, r1098, r1113;
}
{
add.f16x2 r1119, r964, r965;
}
{
mul.f16x2 r1122, r1119, r943;
}
{
add.f16x2 r1125, r967, r1122;
}
{
add.f16x2 r1128, r970, r971;
}
{
mul.f16x2 r1131, r1128, r945;
}
{
add.f16x2 r1134, r1125, r1131;
}
{
sub.f16x2 r1137, r952, r953;
}
{
mul.f16x2 r1140, r1137, r944;
}
{
sub.f16x2 r1143, r958, r959;
}
{
mul.f16x2 r1146, r1143, r946;
}
{
add.f16x2 r1149, r1140, r1146;
}
{
add.f16x2 r1152, r1134, r1149;
}
{
add.f16x2 r1155, r964, r965;
}
{
mul.f16x2 r1158, r1155, r943;
}
{
add.f16x2 r1161, r967, r1158;
}
{
add.f16x2 r1164, r970, r971;
}
{
mul.f16x2 r1167, r1164, r945;
}
{
add.f16x2 r1170, r1161, r1167;
}
{
sub.f16x2 r1173, r952, r953;
}
{
mul.f16x2 r1176, r1173, r944;
}
{
sub.f16x2 r1179, r958, r959;
}
{
mul.f16x2 r1182, r1179, r946;
}
{
add.f16x2 r1185, r1176, r1182;
}
{
sub.f16x2 r1188, r1170, r1185;
}
{
add.f16x2 r1191, r964, r965;
}
{
mul.f16x2 r1194, r1191, r945;
}
{
add.f16x2 r1197, r967, r1194;
}
{
add.f16x2 r1200, r970, r971;
}
{
mul.f16x2 r1203, r1200, r947;
}
{
add.f16x2 r1206, r1197, r1203;
}
{
sub.f16x2 r1209, r952, r953;
}
{
mul.f16x2 r1212, r1209, r946;
}
{
sub.f16x2 r1215, r958, r959;
}
{
mul.f16x2 r1218, r1215, r949;
}
{
add.f16x2 r1221, r1212, r1218;
}
{
add.f16x2 r1224, r1206, r1221;
}
{
add.f16x2 r1227, r964, r965;
}
{
mul.f16x2 r1230, r1227, r945;
}
{
add.f16x2 r1233, r967, r1230;
}
{
add.f16x2 r1236, r970, r971;
}
{
mul.f16x2 r1239, r1236, r947;
}
{
add.f16x2 r1242, r1233, r1239;
}
{
sub.f16x2 r1245, r952, r953;
}
{
mul.f16x2 r1248, r1245, r946;
}
{
sub.f16x2 r1251, r958, r959;
}
{
mul.f16x2 r1254, r1251, r949;
}
{
add.f16x2 r1257, r1248, r1254;
}
{
sub.f16x2 r1260, r1242, r1257;
}
mul.wide.u32 rd6, r2208, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r2222, rd7;
cvt.rn.f32.u32 f115, r2222;
mul.f32 f116, f115, 0f3D4DE32E;
cos.approx.f32 f61, f116;
sin.approx.f32 f117, f116;
neg.f32 f62, f117;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1263, {low, high};
}
mul.lo.s32 r2223, r2222, 25;
sub.s32 r2224, r2208, r2223;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1266, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1268, {high, high};
}
{
mul.f16x2 r1270, r1152, r1268;
}
{
neg.f16x2 r1273, r1270;
}
{
fma.rn.f16x2 r1275, r1008, r1266, r1273;
}
{
mul.f16x2 r1279, r1008, r1268;
}
{
fma.rn.f16x2 r1282, r1152, r1266, r1279;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1286, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1288, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1290, {low, high};
}
{
mul.f16x2 r1291, r1288, r1290;
}
{
mul.f16x2 r1294, r1263, r1286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1297, {high, low};
}
{
fma.rn.f16x2 r1299, r1291, r1297, r1294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1303, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1305, {high, high};
}
{
mul.f16x2 r1307, r1224, r1305;
}
{
neg.f16x2 r1310, r1307;
}
{
fma.rn.f16x2 r1312, r1080, r1303, r1310;
}
{
mul.f16x2 r1316, r1080, r1305;
}
{
fma.rn.f16x2 r1319, r1224, r1303, r1316;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1323, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1325, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1327, {low, high};
}
{
mul.f16x2 r1328, r1325, r1327;
}
{
mul.f16x2 r1331, r1299, r1323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1334, {high, low};
}
{
fma.rn.f16x2 r1336, r1328, r1334, r1331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1340, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1342, {high, high};
}
{
mul.f16x2 r1344, r1260, r1342;
}
{
neg.f16x2 r1347, r1344;
}
{
fma.rn.f16x2 r1349, r1116, r1340, r1347;
}
{
mul.f16x2 r1353, r1116, r1342;
}
{
fma.rn.f16x2 r1356, r1260, r1340, r1353;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1360, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1362, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1364, {low, high};
}
{
mul.f16x2 r1365, r1362, r1364;
}
{
mul.f16x2 r1368, r1336, r1360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1371, {high, low};
}
{
fma.rn.f16x2 r1373, r1365, r1371, r1368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1373;
mov.b32 r1377, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1373;
mov.b32 r1379, {high, high};
}
{
mul.f16x2 r1381, r1188, r1379;
}
{
neg.f16x2 r1384, r1381;
}
{
fma.rn.f16x2 r1386, r1044, r1377, r1384;
}
{
mul.f16x2 r1390, r1044, r1379;
}
{
fma.rn.f16x2 r1393, r1188, r1377, r1390;
}
shl.b32 r2225, r2224, 3;
add.s32 r2226, r2212, r2225;
barrier.sync 0;
mad.lo.s32 r2227, r2222, 1000, r2226;
st.shared.u32 [r2227], r960;
st.shared.u32 [r2227+4], r972;
st.shared.u32 [r2227+200], r1275;
st.shared.u32 [r2227+204], r1282;
st.shared.u32 [r2227+400], r1312;
st.shared.u32 [r2227+404], r1319;
st.shared.u32 [r2227+600], r1349;
st.shared.u32 [r2227+604], r1356;
st.shared.u32 [r2227+800], r1386;
st.shared.u32 [r2227+804], r1393;
barrier.sync 0;
ld.shared.u32 r1426, [r2215];
ld.shared.u32 r1438, [r2215+4];
ld.shared.u32 r1423, [r2215+5000];
ld.shared.u32 r1435, [r2215+5004];
ld.shared.u32 r1429, [r2215+10000];
ld.shared.u32 r1441, [r2215+10004];
ld.shared.u32 r1430, [r2215+15000];
ld.shared.u32 r1442, [r2215+15004];
ld.shared.u32 r1424, [r2215+20000];
ld.shared.u32 r1436, [r2215+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1414, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1417, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1419, {low, high};
}
{
neg.f16x2 r1420, r1419;
}
{
add.f16x2 r1422, r1423, r1424;
}
{
add.f16x2 r1425, r1426, r1422;
}
{
add.f16x2 r1428, r1429, r1430;
}
{
add.f16x2 r1431, r1425, r1428;
}
{
add.f16x2 r1434, r1435, r1436;
}
{
add.f16x2 r1437, r1438, r1434;
}
{
add.f16x2 r1440, r1441, r1442;
}
{
add.f16x2 r1443, r1437, r1440;
}
{
add.f16x2 r1446, r1423, r1424;
}
{
mul.f16x2 r1449, r1446, r1414;
}
{
add.f16x2 r1452, r1426, r1449;
}
{
add.f16x2 r1455, r1429, r1430;
}
{
mul.f16x2 r1458, r1455, r1416;
}
{
add.f16x2 r1461, r1452, r1458;
}
{
sub.f16x2 r1464, r1435, r1436;
}
{
mul.f16x2 r1467, r1464, r1415;
}
{
sub.f16x2 r1470, r1441, r1442;
}
{
mul.f16x2 r1473, r1470, r1417;
}
{
add.f16x2 r1476, r1467, r1473;
}
{
sub.f16x2 r1479, r1461, r1476;
}
{
add.f16x2 r1482, r1423, r1424;
}
{
mul.f16x2 r1485, r1482, r1414;
}
{
add.f16x2 r1488, r1426, r1485;
}
{
add.f16x2 r1491, r1429, r1430;
}
{
mul.f16x2 r1494, r1491, r1416;
}
{
add.f16x2 r1497, r1488, r1494;
}
{
sub.f16x2 r1500, r1435, r1436;
}
{
mul.f16x2 r1503, r1500, r1415;
}
{
sub.f16x2 r1506, r1441, r1442;
}
{
mul.f16x2 r1509, r1506, r1417;
}
{
add.f16x2 r1512, r1503, r1509;
}
{
add.f16x2 r1515, r1497, r1512;
}
{
add.f16x2 r1518, r1423, r1424;
}
{
mul.f16x2 r1521, r1518, r1416;
}
{
add.f16x2 r1524, r1426, r1521;
}
{
add.f16x2 r1527, r1429, r1430;
}
{
mul.f16x2 r1530, r1527, r1418;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
sub.f16x2 r1536, r1435, r1436;
}
{
mul.f16x2 r1539, r1536, r1417;
}
{
sub.f16x2 r1542, r1441, r1442;
}
{
mul.f16x2 r1545, r1542, r1420;
}
{
add.f16x2 r1548, r1539, r1545;
}
{
sub.f16x2 r1551, r1533, r1548;
}
{
add.f16x2 r1554, r1423, r1424;
}
{
mul.f16x2 r1557, r1554, r1416;
}
{
add.f16x2 r1560, r1426, r1557;
}
{
add.f16x2 r1563, r1429, r1430;
}
{
mul.f16x2 r1566, r1563, r1418;
}
{
add.f16x2 r1569, r1560, r1566;
}
{
sub.f16x2 r1572, r1435, r1436;
}
{
mul.f16x2 r1575, r1572, r1417;
}
{
sub.f16x2 r1578, r1441, r1442;
}
{
mul.f16x2 r1581, r1578, r1420;
}
{
add.f16x2 r1584, r1575, r1581;
}
{
add.f16x2 r1587, r1569, r1584;
}
{
add.f16x2 r1590, r1435, r1436;
}
{
mul.f16x2 r1593, r1590, r1414;
}
{
add.f16x2 r1596, r1438, r1593;
}
{
add.f16x2 r1599, r1441, r1442;
}
{
mul.f16x2 r1602, r1599, r1416;
}
{
add.f16x2 r1605, r1596, r1602;
}
{
sub.f16x2 r1608, r1423, r1424;
}
{
mul.f16x2 r1611, r1608, r1415;
}
{
sub.f16x2 r1614, r1429, r1430;
}
{
mul.f16x2 r1617, r1614, r1417;
}
{
add.f16x2 r1620, r1611, r1617;
}
{
add.f16x2 r1623, r1605, r1620;
}
{
add.f16x2 r1626, r1435, r1436;
}
{
mul.f16x2 r1629, r1626, r1414;
}
{
add.f16x2 r1632, r1438, r1629;
}
{
add.f16x2 r1635, r1441, r1442;
}
{
mul.f16x2 r1638, r1635, r1416;
}
{
add.f16x2 r1641, r1632, r1638;
}
{
sub.f16x2 r1644, r1423, r1424;
}
{
mul.f16x2 r1647, r1644, r1415;
}
{
sub.f16x2 r1650, r1429, r1430;
}
{
mul.f16x2 r1653, r1650, r1417;
}
{
add.f16x2 r1656, r1647, r1653;
}
{
sub.f16x2 r1659, r1641, r1656;
}
{
add.f16x2 r1662, r1435, r1436;
}
{
mul.f16x2 r1665, r1662, r1416;
}
{
add.f16x2 r1668, r1438, r1665;
}
{
add.f16x2 r1671, r1441, r1442;
}
{
mul.f16x2 r1674, r1671, r1418;
}
{
add.f16x2 r1677, r1668, r1674;
}
{
sub.f16x2 r1680, r1423, r1424;
}
{
mul.f16x2 r1683, r1680, r1417;
}
{
sub.f16x2 r1686, r1429, r1430;
}
{
mul.f16x2 r1689, r1686, r1420;
}
{
add.f16x2 r1692, r1683, r1689;
}
{
add.f16x2 r1695, r1677, r1692;
}
{
add.f16x2 r1698, r1435, r1436;
}
{
mul.f16x2 r1701, r1698, r1416;
}
{
add.f16x2 r1704, r1438, r1701;
}
{
add.f16x2 r1707, r1441, r1442;
}
{
mul.f16x2 r1710, r1707, r1418;
}
{
add.f16x2 r1713, r1704, r1710;
}
{
sub.f16x2 r1716, r1423, r1424;
}
{
mul.f16x2 r1719, r1716, r1417;
}
{
sub.f16x2 r1722, r1429, r1430;
}
{
mul.f16x2 r1725, r1722, r1420;
}
{
add.f16x2 r1728, r1719, r1725;
}
{
sub.f16x2 r1731, r1713, r1728;
}
mul.wide.u32 rd8, r2208, 274877907;
shr.u64 rd9, rd8, 35;
cvt.u32.u64 r2228, rd9;
cvt.rn.f32.u32 f118, r2228;
mul.f32 f119, f118, 0f3E80ADFD;
cos.approx.f32 f85, f119;
sin.approx.f32 f120, f119;
neg.f32 f86, f120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f85;
cvt.rn.f16.f32 high, f86;
mov.b32 r1734, {low, high};
}
mul.lo.s32 r2229, r2228, 125;
sub.s32 r2230, r2208, r2229;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1739, {high, high};
}
{
mul.f16x2 r1741, r1623, r1739;
}
{
neg.f16x2 r1744, r1741;
}
{
fma.rn.f16x2 r1746, r1479, r1737, r1744;
}
{
mul.f16x2 r1750, r1479, r1739;
}
{
fma.rn.f16x2 r1753, r1623, r1737, r1750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1759, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1761, {low, high};
}
{
mul.f16x2 r1762, r1759, r1761;
}
{
mul.f16x2 r1765, r1734, r1757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1768, {high, low};
}
{
fma.rn.f16x2 r1770, r1762, r1768, r1765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1774, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1776, {high, high};
}
{
mul.f16x2 r1778, r1695, r1776;
}
{
neg.f16x2 r1781, r1778;
}
{
fma.rn.f16x2 r1783, r1551, r1774, r1781;
}
{
mul.f16x2 r1787, r1551, r1776;
}
{
fma.rn.f16x2 r1790, r1695, r1774, r1787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1796, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1798, {low, high};
}
{
mul.f16x2 r1799, r1796, r1798;
}
{
mul.f16x2 r1802, r1770, r1794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1805, {high, low};
}
{
fma.rn.f16x2 r1807, r1799, r1805, r1802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1811, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1813, {high, high};
}
{
mul.f16x2 r1815, r1731, r1813;
}
{
neg.f16x2 r1818, r1815;
}
{
fma.rn.f16x2 r1820, r1587, r1811, r1818;
}
{
mul.f16x2 r1824, r1587, r1813;
}
{
fma.rn.f16x2 r1827, r1731, r1811, r1824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1833, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1835, {low, high};
}
{
mul.f16x2 r1836, r1833, r1835;
}
{
mul.f16x2 r1839, r1807, r1831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1842, {high, low};
}
{
fma.rn.f16x2 r1844, r1836, r1842, r1839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1844;
mov.b32 r1848, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1844;
mov.b32 r1850, {high, high};
}
{
mul.f16x2 r1852, r1659, r1850;
}
{
neg.f16x2 r1855, r1852;
}
{
fma.rn.f16x2 r1857, r1515, r1848, r1855;
}
{
mul.f16x2 r1861, r1515, r1850;
}
{
fma.rn.f16x2 r1864, r1659, r1848, r1861;
}
shl.b32 r2231, r2230, 3;
add.s32 r2232, r2212, r2231;
barrier.sync 0;
mad.lo.s32 r2233, r2228, 5000, r2232;
st.shared.u32 [r2233], r1431;
st.shared.u32 [r2233+4], r1443;
st.shared.u32 [r2233+1000], r1746;
st.shared.u32 [r2233+1004], r1753;
st.shared.u32 [r2233+2000], r1783;
st.shared.u32 [r2233+2004], r1790;
st.shared.u32 [r2233+3000], r1820;
st.shared.u32 [r2233+3004], r1827;
st.shared.u32 [r2233+4000], r1857;
st.shared.u32 [r2233+4004], r1864;
barrier.sync 0;
ld.shared.u32 r1897, [r2215];
ld.shared.u32 r1909, [r2215+4];
ld.shared.u32 r1894, [r2215+5000];
ld.shared.u32 r1906, [r2215+5004];
ld.shared.u32 r1900, [r2215+10000];
ld.shared.u32 r1912, [r2215+10004];
ld.shared.u32 r1901, [r2215+15000];
ld.shared.u32 r1913, [r2215+15004];
ld.shared.u32 r1895, [r2215+20000];
ld.shared.u32 r1907, [r2215+20004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1886, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1887, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1888, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1890, {low, high};
}
{
neg.f16x2 r1891, r1890;
}
{
add.f16x2 r1893, r1894, r1895;
}
{
add.f16x2 r1896, r1897, r1893;
}
{
add.f16x2 r1899, r1900, r1901;
}
{
add.f16x2 %0, r1896, r1899;
}
{
add.f16x2 r1905, r1906, r1907;
}
{
add.f16x2 r1908, r1909, r1905;
}
{
add.f16x2 r1911, r1912, r1913;
}
{
add.f16x2 %1, r1908, r1911;
}
{
add.f16x2 r1917, r1894, r1895;
}
{
mul.f16x2 r1920, r1917, r1885;
}
{
add.f16x2 r1923, r1897, r1920;
}
{
add.f16x2 r1926, r1900, r1901;
}
{
mul.f16x2 r1929, r1926, r1887;
}
{
add.f16x2 r1932, r1923, r1929;
}
{
sub.f16x2 r1935, r1906, r1907;
}
{
mul.f16x2 r1938, r1935, r1886;
}
{
sub.f16x2 r1941, r1912, r1913;
}
{
mul.f16x2 r1944, r1941, r1888;
}
{
add.f16x2 r1947, r1938, r1944;
}
{
sub.f16x2 %2, r1932, r1947;
}
{
add.f16x2 r1953, r1894, r1895;
}
{
mul.f16x2 r1956, r1953, r1885;
}
{
add.f16x2 r1959, r1897, r1956;
}
{
add.f16x2 r1962, r1900, r1901;
}
{
mul.f16x2 r1965, r1962, r1887;
}
{
add.f16x2 r1968, r1959, r1965;
}
{
sub.f16x2 r1971, r1906, r1907;
}
{
mul.f16x2 r1974, r1971, r1886;
}
{
sub.f16x2 r1977, r1912, r1913;
}
{
mul.f16x2 r1980, r1977, r1888;
}
{
add.f16x2 r1983, r1974, r1980;
}
{
add.f16x2 %8, r1968, r1983;
}
{
add.f16x2 r1989, r1894, r1895;
}
{
mul.f16x2 r1992, r1989, r1887;
}
{
add.f16x2 r1995, r1897, r1992;
}
{
add.f16x2 r1998, r1900, r1901;
}
{
mul.f16x2 r2001, r1998, r1889;
}
{
add.f16x2 r2004, r1995, r2001;
}
{
sub.f16x2 r2007, r1906, r1907;
}
{
mul.f16x2 r2010, r2007, r1888;
}
{
sub.f16x2 r2013, r1912, r1913;
}
{
mul.f16x2 r2016, r2013, r1891;
}
{
add.f16x2 r2019, r2010, r2016;
}
{
sub.f16x2 %4, r2004, r2019;
}
{
add.f16x2 r2025, r1894, r1895;
}
{
mul.f16x2 r2028, r2025, r1887;
}
{
add.f16x2 r2031, r1897, r2028;
}
{
add.f16x2 r2034, r1900, r1901;
}
{
mul.f16x2 r2037, r2034, r1889;
}
{
add.f16x2 r2040, r2031, r2037;
}
{
sub.f16x2 r2043, r1906, r1907;
}
{
mul.f16x2 r2046, r2043, r1888;
}
{
sub.f16x2 r2049, r1912, r1913;
}
{
mul.f16x2 r2052, r2049, r1891;
}
{
add.f16x2 r2055, r2046, r2052;
}
{
add.f16x2 %6, r2040, r2055;
}
{
add.f16x2 r2061, r1906, r1907;
}
{
mul.f16x2 r2064, r2061, r1885;
}
{
add.f16x2 r2067, r1909, r2064;
}
{
add.f16x2 r2070, r1912, r1913;
}
{
mul.f16x2 r2073, r2070, r1887;
}
{
add.f16x2 r2076, r2067, r2073;
}
{
sub.f16x2 r2079, r1894, r1895;
}
{
mul.f16x2 r2082, r2079, r1886;
}
{
sub.f16x2 r2085, r1900, r1901;
}
{
mul.f16x2 r2088, r2085, r1888;
}
{
add.f16x2 r2091, r2082, r2088;
}
{
add.f16x2 %3, r2076, r2091;
}
{
add.f16x2 r2097, r1906, r1907;
}
{
mul.f16x2 r2100, r2097, r1885;
}
{
add.f16x2 r2103, r1909, r2100;
}
{
add.f16x2 r2106, r1912, r1913;
}
{
mul.f16x2 r2109, r2106, r1887;
}
{
add.f16x2 r2112, r2103, r2109;
}
{
sub.f16x2 r2115, r1894, r1895;
}
{
mul.f16x2 r2118, r2115, r1886;
}
{
sub.f16x2 r2121, r1900, r1901;
}
{
mul.f16x2 r2124, r2121, r1888;
}
{
add.f16x2 r2127, r2118, r2124;
}
{
sub.f16x2 %9, r2112, r2127;
}
{
add.f16x2 r2133, r1906, r1907;
}
{
mul.f16x2 r2136, r2133, r1887;
}
{
add.f16x2 r2139, r1909, r2136;
}
{
add.f16x2 r2142, r1912, r1913;
}
{
mul.f16x2 r2145, r2142, r1889;
}
{
add.f16x2 r2148, r2139, r2145;
}
{
sub.f16x2 r2151, r1894, r1895;
}
{
mul.f16x2 r2154, r2151, r1888;
}
{
sub.f16x2 r2157, r1900, r1901;
}
{
mul.f16x2 r2160, r2157, r1891;
}
{
add.f16x2 r2163, r2154, r2160;
}
{
add.f16x2 %5, r2148, r2163;
}
{
add.f16x2 r2169, r1906, r1907;
}
{
mul.f16x2 r2172, r2169, r1887;
}
{
add.f16x2 r2175, r1909, r2172;
}
{
add.f16x2 r2178, r1912, r1913;
}
{
mul.f16x2 r2181, r2178, r1889;
}
{
add.f16x2 r2184, r2175, r2181;
}
{
sub.f16x2 r2187, r1894, r1895;
}
{
mul.f16x2 r2190, r2187, r1888;
}
{
sub.f16x2 r2193, r1900, r1901;
}
{
mul.f16x2 r2196, r2193, r1891;
}
{
add.f16x2 r2199, r2190, r2196;
}
{
sub.f16x2 %7, r2184, r2199;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<915, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<121>;
.reg .b32 r<2234>;
.reg .b64 rd<10>;
mov.u32 r2205, %tid.y;
mov.u32 r2206, %10;
mad.lo.s32 r2207, r2205, 12500, r2206;
mov.u32 r2208, %tid.x;
mov.f32 f106, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1, {low, high};
}
mov.f32 f108, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r2, {low, high};
}
mov.f32 f102, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r3, {low, high};
}
mov.f32 f104, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r6, {low, high};
}
{
neg.f16x2 r7, r6;
}
{
add.f16x2 r9, %13, %19;
}
{
add.f16x2 r12, %11, r9;
}
{
add.f16x2 r15, %15, %17;
}
{
add.f16x2 r18, r12, r15;
}
{
add.f16x2 r21, %14, %20;
}
{
add.f16x2 r24, %12, r21;
}
{
add.f16x2 r27, %16, %18;
}
{
add.f16x2 r30, r24, r27;
}
{
add.f16x2 r33, %13, %19;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %11, r36;
}
{
add.f16x2 r42, %15, %17;
}
{
mul.f16x2 r45, r42, r3;
}
{
add.f16x2 r48, r39, r45;
}
{
sub.f16x2 r51, %14, %20;
}
{
mul.f16x2 r54, r51, r2;
}
{
sub.f16x2 r57, %16, %18;
}
{
mul.f16x2 r60, r57, r4;
}
{
add.f16x2 r63, r54, r60;
}
{
sub.f16x2 r66, r48, r63;
}
{
add.f16x2 r69, %13, %19;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %11, r72;
}
{
add.f16x2 r78, %15, %17;
}
{
mul.f16x2 r81, r78, r3;
}
{
add.f16x2 r84, r75, r81;
}
{
sub.f16x2 r87, %14, %20;
}
{
mul.f16x2 r90, r87, r2;
}
{
sub.f16x2 r93, %16, %18;
}
{
mul.f16x2 r96, r93, r4;
}
{
add.f16x2 r99, r90, r96;
}
{
add.f16x2 r102, r84, r99;
}
{
add.f16x2 r105, %13, %19;
}
{
mul.f16x2 r108, r105, r3;
}
{
add.f16x2 r111, %11, r108;
}
{
add.f16x2 r114, %15, %17;
}
{
mul.f16x2 r117, r114, r5;
}
{
add.f16x2 r120, r111, r117;
}
{
sub.f16x2 r123, %14, %20;
}
{
mul.f16x2 r126, r123, r4;
}
{
sub.f16x2 r129, %16, %18;
}
{
mul.f16x2 r132, r129, r7;
}
{
add.f16x2 r135, r126, r132;
}
{
sub.f16x2 r138, r120, r135;
}
{
add.f16x2 r141, %13, %19;
}
{
mul.f16x2 r144, r141, r3;
}
{
add.f16x2 r147, %11, r144;
}
{
add.f16x2 r150, %15, %17;
}
{
mul.f16x2 r153, r150, r5;
}
{
add.f16x2 r156, r147, r153;
}
{
sub.f16x2 r159, %14, %20;
}
{
mul.f16x2 r162, r159, r4;
}
{
sub.f16x2 r165, %16, %18;
}
{
mul.f16x2 r168, r165, r7;
}
{
add.f16x2 r171, r162, r168;
}
{
add.f16x2 r174, r156, r171;
}
{
add.f16x2 r177, %14, %20;
}
{
mul.f16x2 r180, r177, r1;
}
{
add.f16x2 r183, %12, r180;
}
{
add.f16x2 r186, %16, %18;
}
{
mul.f16x2 r189, r186, r3;
}
{
add.f16x2 r192, r183, r189;
}
{
sub.f16x2 r195, %13, %19;
}
{
mul.f16x2 r198, r195, r2;
}
{
sub.f16x2 r201, %15, %17;
}
{
mul.f16x2 r204, r201, r4;
}
{
add.f16x2 r207, r198, r204;
}
{
add.f16x2 r210, r192, r207;
}
{
add.f16x2 r213, %14, %20;
}
{
mul.f16x2 r216, r213, r1;
}
{
add.f16x2 r219, %12, r216;
}
{
add.f16x2 r222, %16, %18;
}
{
mul.f16x2 r225, r222, r3;
}
{
add.f16x2 r228, r219, r225;
}
{
sub.f16x2 r231, %13, %19;
}
{
mul.f16x2 r234, r231, r2;
}
{
sub.f16x2 r237, %15, %17;
}
{
mul.f16x2 r240, r237, r4;
}
{
add.f16x2 r243, r234, r240;
}
{
sub.f16x2 r246, r228, r243;
}
{
add.f16x2 r249, %14, %20;
}
{
mul.f16x2 r252, r249, r3;
}
{
add.f16x2 r255, %12, r252;
}
{
add.f16x2 r258, %16, %18;
}
{
mul.f16x2 r261, r258, r5;
}
{
add.f16x2 r264, r255, r261;
}
{
sub.f16x2 r267, %13, %19;
}
{
mul.f16x2 r270, r267, r4;
}
{
sub.f16x2 r273, %15, %17;
}
{
mul.f16x2 r276, r273, r7;
}
{
add.f16x2 r279, r270, r276;
}
{
add.f16x2 r282, r264, r279;
}
{
add.f16x2 r285, %14, %20;
}
{
mul.f16x2 r288, r285, r3;
}
{
add.f16x2 r291, %12, r288;
}
{
add.f16x2 r294, %16, %18;
}
{
mul.f16x2 r297, r294, r5;
}
{
add.f16x2 r300, r291, r297;
}
{
sub.f16x2 r303, %13, %19;
}
{
mul.f16x2 r306, r303, r4;
}
{
sub.f16x2 r309, %15, %17;
}
{
mul.f16x2 r312, r309, r7;
}
{
add.f16x2 r315, r306, r312;
}
{
sub.f16x2 r318, r300, r315;
}
mul.wide.u32 rd2, r2208, -776530087;
shr.u64 rd3, rd2, 41;
cvt.u32.u64 r2209, rd3;
mul.lo.s32 r2210, r2209, 625;
sub.s32 r2211, r2208, r2210;
cvt.rn.f32.u32 f109, r2211;
mul.f32 f110, f109, 0f3B03C498;
cos.approx.f32 f13, f110;
sin.approx.f32 f111, f110;
neg.f32 f14, f111;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r324, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r326, {high, high};
}
{
mul.f16x2 r328, r210, r326;
}
{
neg.f16x2 r331, r328;
}
{
fma.rn.f16x2 r333, r66, r324, r331;
}
{
mul.f16x2 r337, r66, r326;
}
{
fma.rn.f16x2 r340, r210, r324, r337;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r344, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r346, {high, high};
}
mov.f32 f93, 0fBF800000;
mov.f32 f94, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r348, {low, high};
}
{
mul.f16x2 r349, r346, r348;
}
{
mul.f16x2 r352, r321, r344;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r355, {high, low};
}
{
fma.rn.f16x2 r357, r349, r355, r352;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r361, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r363, {high, high};
}
{
mul.f16x2 r365, r282, r363;
}
{
neg.f16x2 r368, r365;
}
{
fma.rn.f16x2 r370, r138, r361, r368;
}
{
mul.f16x2 r374, r138, r363;
}
{
fma.rn.f16x2 r377, r282, r361, r374;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r381, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r383, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r385, {low, high};
}
{
mul.f16x2 r386, r383, r385;
}
{
mul.f16x2 r389, r357, r381;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r357;
mov.b32 r392, {high, low};
}
{
fma.rn.f16x2 r394, r386, r392, r389;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r398, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r400, {high, high};
}
{
mul.f16x2 r402, r318, r400;
}
{
neg.f16x2 r405, r402;
}
{
fma.rn.f16x2 r407, r174, r398, r405;
}
{
mul.f16x2 r411, r174, r400;
}
{
fma.rn.f16x2 r414, r318, r398, r411;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r418, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r321;
mov.b32 r420, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r422, {low, high};
}
{
mul.f16x2 r423, r420, r422;
}
{
mul.f16x2 r426, r394, r418;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r394;
mov.b32 r429, {high, low};
}
{
fma.rn.f16x2 r431, r423, r429, r426;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r431;
mov.b32 r435, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r431;
mov.b32 r437, {high, high};
}
{
mul.f16x2 r439, r246, r437;
}
{
neg.f16x2 r442, r439;
}
{
fma.rn.f16x2 r444, r102, r435, r442;
}
{
mul.f16x2 r448, r102, r437;
}
{
fma.rn.f16x2 r451, r246, r435, r448;
}
mad.lo.s32 r2212, r2209, 12500, r2207;
barrier.sync 0;
mad.lo.s32 r2213, r2211, 20, r2212;
st.shared.u32 [r2213], r18;
st.shared.u32 [r2213+4], r333;
st.shared.u32 [r2213+8], r370;
st.shared.u32 [r2213+12], r407;
st.shared.u32 [r2213+16], r444;
barrier.sync 0;
shl.b32 r2214, r2211, 4;
sub.s32 r2215, r2213, r2214;
ld.shared.u32 r484, [r2215];
ld.shared.u32 r481, [r2215+2500];
ld.shared.u32 r487, [r2215+5000];
ld.shared.u32 r488, [r2215+7500];
ld.shared.u32 r482, [r2215+10000];
barrier.sync 0;
st.shared.u32 [r2213], r30;
st.shared.u32 [r2213+4], r340;
st.shared.u32 [r2213+8], r377;
st.shared.u32 [r2213+12], r414;
st.shared.u32 [r2213+16], r451;
barrier.sync 0;
ld.shared.u32 r496, [r2215];
ld.shared.u32 r493, [r2215+2500];
ld.shared.u32 r499, [r2215+5000];
ld.shared.u32 r500, [r2215+7500];
ld.shared.u32 r494, [r2215+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r477, {low, high};
}
{
neg.f16x2 r478, r477;
}
{
add.f16x2 r480, r481, r482;
}
{
add.f16x2 r483, r484, r480;
}
{
add.f16x2 r486, r487, r488;
}
{
add.f16x2 r489, r483, r486;
}
{
add.f16x2 r492, r493, r494;
}
{
add.f16x2 r495, r496, r492;
}
{
add.f16x2 r498, r499, r500;
}
{
add.f16x2 r501, r495, r498;
}
{
add.f16x2 r504, r481, r482;
}
{
mul.f16x2 r507, r504, r472;
}
{
add.f16x2 r510, r484, r507;
}
{
add.f16x2 r513, r487, r488;
}
{
mul.f16x2 r516, r513, r474;
}
{
add.f16x2 r519, r510, r516;
}
{
sub.f16x2 r522, r493, r494;
}
{
mul.f16x2 r525, r522, r473;
}
{
sub.f16x2 r528, r499, r500;
}
{
mul.f16x2 r531, r528, r475;
}
{
add.f16x2 r534, r525, r531;
}
{
sub.f16x2 r537, r519, r534;
}
{
add.f16x2 r540, r481, r482;
}
{
mul.f16x2 r543, r540, r472;
}
{
add.f16x2 r546, r484, r543;
}
{
add.f16x2 r549, r487, r488;
}
{
mul.f16x2 r552, r549, r474;
}
{
add.f16x2 r555, r546, r552;
}
{
sub.f16x2 r558, r493, r494;
}
{
mul.f16x2 r561, r558, r473;
}
{
sub.f16x2 r564, r499, r500;
}
{
mul.f16x2 r567, r564, r475;
}
{
add.f16x2 r570, r561, r567;
}
{
add.f16x2 r573, r555, r570;
}
{
add.f16x2 r576, r481, r482;
}
{
mul.f16x2 r579, r576, r474;
}
{
add.f16x2 r582, r484, r579;
}
{
add.f16x2 r585, r487, r488;
}
{
mul.f16x2 r588, r585, r476;
}
{
add.f16x2 r591, r582, r588;
}
{
sub.f16x2 r594, r493, r494;
}
{
mul.f16x2 r597, r594, r475;
}
{
sub.f16x2 r600, r499, r500;
}
{
mul.f16x2 r603, r600, r478;
}
{
add.f16x2 r606, r597, r603;
}
{
sub.f16x2 r609, r591, r606;
}
{
add.f16x2 r612, r481, r482;
}
{
mul.f16x2 r615, r612, r474;
}
{
add.f16x2 r618, r484, r615;
}
{
add.f16x2 r621, r487, r488;
}
{
mul.f16x2 r624, r621, r476;
}
{
add.f16x2 r627, r618, r624;
}
{
sub.f16x2 r630, r493, r494;
}
{
mul.f16x2 r633, r630, r475;
}
{
sub.f16x2 r636, r499, r500;
}
{
mul.f16x2 r639, r636, r478;
}
{
add.f16x2 r642, r633, r639;
}
{
add.f16x2 r645, r627, r642;
}
{
add.f16x2 r648, r493, r494;
}
{
mul.f16x2 r651, r648, r472;
}
{
add.f16x2 r654, r496, r651;
}
{
add.f16x2 r657, r499, r500;
}
{
mul.f16x2 r660, r657, r474;
}
{
add.f16x2 r663, r654, r660;
}
{
sub.f16x2 r666, r481, r482;
}
{
mul.f16x2 r669, r666, r473;
}
{
sub.f16x2 r672, r487, r488;
}
{
mul.f16x2 r675, r672, r475;
}
{
add.f16x2 r678, r669, r675;
}
{
add.f16x2 r681, r663, r678;
}
{
add.f16x2 r684, r493, r494;
}
{
mul.f16x2 r687, r684, r472;
}
{
add.f16x2 r690, r496, r687;
}
{
add.f16x2 r693, r499, r500;
}
{
mul.f16x2 r696, r693, r474;
}
{
add.f16x2 r699, r690, r696;
}
{
sub.f16x2 r702, r481, r482;
}
{
mul.f16x2 r705, r702, r473;
}
{
sub.f16x2 r708, r487, r488;
}
{
mul.f16x2 r711, r708, r475;
}
{
add.f16x2 r714, r705, r711;
}
{
sub.f16x2 r717, r699, r714;
}
{
add.f16x2 r720, r493, r494;
}
{
mul.f16x2 r723, r720, r474;
}
{
add.f16x2 r726, r496, r723;
}
{
add.f16x2 r729, r499, r500;
}
{
mul.f16x2 r732, r729, r476;
}
{
add.f16x2 r735, r726, r732;
}
{
sub.f16x2 r738, r481, r482;
}
{
mul.f16x2 r741, r738, r475;
}
{
sub.f16x2 r744, r487, r488;
}
{
mul.f16x2 r747, r744, r478;
}
{
add.f16x2 r750, r741, r747;
}
{
add.f16x2 r753, r735, r750;
}
{
add.f16x2 r756, r493, r494;
}
{
mul.f16x2 r759, r756, r474;
}
{
add.f16x2 r762, r496, r759;
}
{
add.f16x2 r765, r499, r500;
}
{
mul.f16x2 r768, r765, r476;
}
{
add.f16x2 r771, r762, r768;
}
{
sub.f16x2 r774, r481, r482;
}
{
mul.f16x2 r777, r774, r475;
}
{
sub.f16x2 r780, r487, r488;
}
{
mul.f16x2 r783, r780, r478;
}
{
add.f16x2 r786, r777, r783;
}
{
sub.f16x2 r789, r771, r786;
}
mul.wide.u32 rd4, r2211, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r2216, rd5;
cvt.rn.f32.u32 f112, r2216;
mul.f32 f113, f112, 0f3C24B5BE;
cos.approx.f32 f37, f113;
sin.approx.f32 f114, f113;
neg.f32 f38, f114;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r792, {low, high};
}
mul.lo.s32 r2217, r2216, 5;
sub.s32 r2218, r2211, r2217;
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r795, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r797, {high, high};
}
{
mul.f16x2 r799, r681, r797;
}
{
neg.f16x2 r802, r799;
}
{
fma.rn.f16x2 r804, r537, r795, r802;
}
{
mul.f16x2 r808, r537, r797;
}
{
fma.rn.f16x2 r811, r681, r795, r808;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r815, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r817, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r819, {low, high};
}
{
mul.f16x2 r820, r817, r819;
}
{
mul.f16x2 r823, r792, r815;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r826, {high, low};
}
{
fma.rn.f16x2 r828, r820, r826, r823;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r832, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r834, {high, high};
}
{
mul.f16x2 r836, r753, r834;
}
{
neg.f16x2 r839, r836;
}
{
fma.rn.f16x2 r841, r609, r832, r839;
}
{
mul.f16x2 r845, r609, r834;
}
{
fma.rn.f16x2 r848, r753, r832, r845;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r852, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r854, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r856, {low, high};
}
{
mul.f16x2 r857, r854, r856;
}
{
mul.f16x2 r860, r828, r852;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r828;
mov.b32 r863, {high, low};
}
{
fma.rn.f16x2 r865, r857, r863, r860;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r869, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r871, {high, high};
}
{
mul.f16x2 r873, r789, r871;
}
{
neg.f16x2 r876, r873;
}
{
fma.rn.f16x2 r878, r645, r869, r876;
}
{
mul.f16x2 r882, r645, r871;
}
{
fma.rn.f16x2 r885, r789, r869, r882;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r889, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r792;
mov.b32 r891, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r893, {low, high};
}
{
mul.f16x2 r894, r891, r893;
}
{
mul.f16x2 r897, r865, r889;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r865;
mov.b32 r900, {high, low};
}
{
fma.rn.f16x2 r902, r894, r900, r897;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r906, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r902;
mov.b32 r908, {high, high};
}
{
mul.f16x2 r910, r717, r908;
}
{
neg.f16x2 r913, r910;
}
{
fma.rn.f16x2 r915, r573, r906, r913;
}
{
mul.f16x2 r919, r573, r908;
}
{
fma.rn.f16x2 r922, r717, r906, r919;
}
shl.b32 r2219, r2218, 2;
add.s32 r2220, r2212, r2219;
barrier.sync 0;
mad.lo.s32 r2221, r2216, 100, r2220;
st.shared.u32 [r2221], r489;
st.shared.u32 [r2221+20], r804;
st.shared.u32 [r2221+40], r841;
st.shared.u32 [r2221+60], r878;
st.shared.u32 [r2221+80], r915;
barrier.sync 0;
ld.shared.u32 r955, [r2215];
ld.shared.u32 r952, [r2215+2500];
ld.shared.u32 r958, [r2215+5000];
ld.shared.u32 r959, [r2215+7500];
ld.shared.u32 r953, [r2215+10000];
barrier.sync 0;
st.shared.u32 [r2221], r501;
st.shared.u32 [r2221+20], r811;
st.shared.u32 [r2221+40], r848;
st.shared.u32 [r2221+60], r885;
st.shared.u32 [r2221+80], r922;
barrier.sync 0;
ld.shared.u32 r967, [r2215];
ld.shared.u32 r964, [r2215+2500];
ld.shared.u32 r970, [r2215+5000];
ld.shared.u32 r971, [r2215+7500];
ld.shared.u32 r965, [r2215+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r946, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r948, {low, high};
}
{
neg.f16x2 r949, r948;
}
{
add.f16x2 r951, r952, r953;
}
{
add.f16x2 r954, r955, r951;
}
{
add.f16x2 r957, r958, r959;
}
{
add.f16x2 r960, r954, r957;
}
{
add.f16x2 r963, r964, r965;
}
{
add.f16x2 r966, r967, r963;
}
{
add.f16x2 r969, r970, r971;
}
{
add.f16x2 r972, r966, r969;
}
{
add.f16x2 r975, r952, r953;
}
{
mul.f16x2 r978, r975, r943;
}
{
add.f16x2 r981, r955, r978;
}
{
add.f16x2 r984, r958, r959;
}
{
mul.f16x2 r987, r984, r945;
}
{
add.f16x2 r990, r981, r987;
}
{
sub.f16x2 r993, r964, r965;
}
{
mul.f16x2 r996, r993, r944;
}
{
sub.f16x2 r999, r970, r971;
}
{
mul.f16x2 r1002, r999, r946;
}
{
add.f16x2 r1005, r996, r1002;
}
{
sub.f16x2 r1008, r990, r1005;
}
{
add.f16x2 r1011, r952, r953;
}
{
mul.f16x2 r1014, r1011, r943;
}
{
add.f16x2 r1017, r955, r1014;
}
{
add.f16x2 r1020, r958, r959;
}
{
mul.f16x2 r1023, r1020, r945;
}
{
add.f16x2 r1026, r1017, r1023;
}
{
sub.f16x2 r1029, r964, r965;
}
{
mul.f16x2 r1032, r1029, r944;
}
{
sub.f16x2 r1035, r970, r971;
}
{
mul.f16x2 r1038, r1035, r946;
}
{
add.f16x2 r1041, r1032, r1038;
}
{
add.f16x2 r1044, r1026, r1041;
}
{
add.f16x2 r1047, r952, r953;
}
{
mul.f16x2 r1050, r1047, r945;
}
{
add.f16x2 r1053, r955, r1050;
}
{
add.f16x2 r1056, r958, r959;
}
{
mul.f16x2 r1059, r1056, r947;
}
{
add.f16x2 r1062, r1053, r1059;
}
{
sub.f16x2 r1065, r964, r965;
}
{
mul.f16x2 r1068, r1065, r946;
}
{
sub.f16x2 r1071, r970, r971;
}
{
mul.f16x2 r1074, r1071, r949;
}
{
add.f16x2 r1077, r1068, r1074;
}
{
sub.f16x2 r1080, r1062, r1077;
}
{
add.f16x2 r1083, r952, r953;
}
{
mul.f16x2 r1086, r1083, r945;
}
{
add.f16x2 r1089, r955, r1086;
}
{
add.f16x2 r1092, r958, r959;
}
{
mul.f16x2 r1095, r1092, r947;
}
{
add.f16x2 r1098, r1089, r1095;
}
{
sub.f16x2 r1101, r964, r965;
}
{
mul.f16x2 r1104, r1101, r946;
}
{
sub.f16x2 r1107, r970, r971;
}
{
mul.f16x2 r1110, r1107, r949;
}
{
add.f16x2 r1113, r1104, r1110;
}
{
add.f16x2 r1116, r1098, r1113;
}
{
add.f16x2 r1119, r964, r965;
}
{
mul.f16x2 r1122, r1119, r943;
}
{
add.f16x2 r1125, r967, r1122;
}
{
add.f16x2 r1128, r970, r971;
}
{
mul.f16x2 r1131, r1128, r945;
}
{
add.f16x2 r1134, r1125, r1131;
}
{
sub.f16x2 r1137, r952, r953;
}
{
mul.f16x2 r1140, r1137, r944;
}
{
sub.f16x2 r1143, r958, r959;
}
{
mul.f16x2 r1146, r1143, r946;
}
{
add.f16x2 r1149, r1140, r1146;
}
{
add.f16x2 r1152, r1134, r1149;
}
{
add.f16x2 r1155, r964, r965;
}
{
mul.f16x2 r1158, r1155, r943;
}
{
add.f16x2 r1161, r967, r1158;
}
{
add.f16x2 r1164, r970, r971;
}
{
mul.f16x2 r1167, r1164, r945;
}
{
add.f16x2 r1170, r1161, r1167;
}
{
sub.f16x2 r1173, r952, r953;
}
{
mul.f16x2 r1176, r1173, r944;
}
{
sub.f16x2 r1179, r958, r959;
}
{
mul.f16x2 r1182, r1179, r946;
}
{
add.f16x2 r1185, r1176, r1182;
}
{
sub.f16x2 r1188, r1170, r1185;
}
{
add.f16x2 r1191, r964, r965;
}
{
mul.f16x2 r1194, r1191, r945;
}
{
add.f16x2 r1197, r967, r1194;
}
{
add.f16x2 r1200, r970, r971;
}
{
mul.f16x2 r1203, r1200, r947;
}
{
add.f16x2 r1206, r1197, r1203;
}
{
sub.f16x2 r1209, r952, r953;
}
{
mul.f16x2 r1212, r1209, r946;
}
{
sub.f16x2 r1215, r958, r959;
}
{
mul.f16x2 r1218, r1215, r949;
}
{
add.f16x2 r1221, r1212, r1218;
}
{
add.f16x2 r1224, r1206, r1221;
}
{
add.f16x2 r1227, r964, r965;
}
{
mul.f16x2 r1230, r1227, r945;
}
{
add.f16x2 r1233, r967, r1230;
}
{
add.f16x2 r1236, r970, r971;
}
{
mul.f16x2 r1239, r1236, r947;
}
{
add.f16x2 r1242, r1233, r1239;
}
{
sub.f16x2 r1245, r952, r953;
}
{
mul.f16x2 r1248, r1245, r946;
}
{
sub.f16x2 r1251, r958, r959;
}
{
mul.f16x2 r1254, r1251, r949;
}
{
add.f16x2 r1257, r1248, r1254;
}
{
sub.f16x2 r1260, r1242, r1257;
}
mul.wide.u32 rd6, r2211, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r2222, rd7;
cvt.rn.f32.u32 f115, r2222;
mul.f32 f116, f115, 0f3D4DE32E;
cos.approx.f32 f61, f116;
sin.approx.f32 f117, f116;
neg.f32 f62, f117;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1263, {low, high};
}
mul.lo.s32 r2223, r2222, 25;
sub.s32 r2224, r2211, r2223;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1266, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1268, {high, high};
}
{
mul.f16x2 r1270, r1152, r1268;
}
{
neg.f16x2 r1273, r1270;
}
{
fma.rn.f16x2 r1275, r1008, r1266, r1273;
}
{
mul.f16x2 r1279, r1008, r1268;
}
{
fma.rn.f16x2 r1282, r1152, r1266, r1279;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1286, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1288, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1290, {low, high};
}
{
mul.f16x2 r1291, r1288, r1290;
}
{
mul.f16x2 r1294, r1263, r1286;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1297, {high, low};
}
{
fma.rn.f16x2 r1299, r1291, r1297, r1294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1303, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1305, {high, high};
}
{
mul.f16x2 r1307, r1224, r1305;
}
{
neg.f16x2 r1310, r1307;
}
{
fma.rn.f16x2 r1312, r1080, r1303, r1310;
}
{
mul.f16x2 r1316, r1080, r1305;
}
{
fma.rn.f16x2 r1319, r1224, r1303, r1316;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1323, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1325, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1327, {low, high};
}
{
mul.f16x2 r1328, r1325, r1327;
}
{
mul.f16x2 r1331, r1299, r1323;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1299;
mov.b32 r1334, {high, low};
}
{
fma.rn.f16x2 r1336, r1328, r1334, r1331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1340, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1342, {high, high};
}
{
mul.f16x2 r1344, r1260, r1342;
}
{
neg.f16x2 r1347, r1344;
}
{
fma.rn.f16x2 r1349, r1116, r1340, r1347;
}
{
mul.f16x2 r1353, r1116, r1342;
}
{
fma.rn.f16x2 r1356, r1260, r1340, r1353;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1360, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1263;
mov.b32 r1362, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1364, {low, high};
}
{
mul.f16x2 r1365, r1362, r1364;
}
{
mul.f16x2 r1368, r1336, r1360;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1336;
mov.b32 r1371, {high, low};
}
{
fma.rn.f16x2 r1373, r1365, r1371, r1368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1373;
mov.b32 r1377, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1373;
mov.b32 r1379, {high, high};
}
{
mul.f16x2 r1381, r1188, r1379;
}
{
neg.f16x2 r1384, r1381;
}
{
fma.rn.f16x2 r1386, r1044, r1377, r1384;
}
{
mul.f16x2 r1390, r1044, r1379;
}
{
fma.rn.f16x2 r1393, r1188, r1377, r1390;
}
shl.b32 r2225, r2224, 2;
add.s32 r2226, r2212, r2225;
barrier.sync 0;
mad.lo.s32 r2227, r2222, 500, r2226;
st.shared.u32 [r2227], r960;
st.shared.u32 [r2227+100], r1275;
st.shared.u32 [r2227+200], r1312;
st.shared.u32 [r2227+300], r1349;
st.shared.u32 [r2227+400], r1386;
barrier.sync 0;
ld.shared.u32 r1426, [r2215];
ld.shared.u32 r1423, [r2215+2500];
ld.shared.u32 r1429, [r2215+5000];
ld.shared.u32 r1430, [r2215+7500];
ld.shared.u32 r1424, [r2215+10000];
barrier.sync 0;
st.shared.u32 [r2227], r972;
st.shared.u32 [r2227+100], r1282;
st.shared.u32 [r2227+200], r1319;
st.shared.u32 [r2227+300], r1356;
st.shared.u32 [r2227+400], r1393;
barrier.sync 0;
ld.shared.u32 r1438, [r2215];
ld.shared.u32 r1435, [r2215+2500];
ld.shared.u32 r1441, [r2215+5000];
ld.shared.u32 r1442, [r2215+7500];
ld.shared.u32 r1436, [r2215+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1414, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1417, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1419, {low, high};
}
{
neg.f16x2 r1420, r1419;
}
{
add.f16x2 r1422, r1423, r1424;
}
{
add.f16x2 r1425, r1426, r1422;
}
{
add.f16x2 r1428, r1429, r1430;
}
{
add.f16x2 r1431, r1425, r1428;
}
{
add.f16x2 r1434, r1435, r1436;
}
{
add.f16x2 r1437, r1438, r1434;
}
{
add.f16x2 r1440, r1441, r1442;
}
{
add.f16x2 r1443, r1437, r1440;
}
{
add.f16x2 r1446, r1423, r1424;
}
{
mul.f16x2 r1449, r1446, r1414;
}
{
add.f16x2 r1452, r1426, r1449;
}
{
add.f16x2 r1455, r1429, r1430;
}
{
mul.f16x2 r1458, r1455, r1416;
}
{
add.f16x2 r1461, r1452, r1458;
}
{
sub.f16x2 r1464, r1435, r1436;
}
{
mul.f16x2 r1467, r1464, r1415;
}
{
sub.f16x2 r1470, r1441, r1442;
}
{
mul.f16x2 r1473, r1470, r1417;
}
{
add.f16x2 r1476, r1467, r1473;
}
{
sub.f16x2 r1479, r1461, r1476;
}
{
add.f16x2 r1482, r1423, r1424;
}
{
mul.f16x2 r1485, r1482, r1414;
}
{
add.f16x2 r1488, r1426, r1485;
}
{
add.f16x2 r1491, r1429, r1430;
}
{
mul.f16x2 r1494, r1491, r1416;
}
{
add.f16x2 r1497, r1488, r1494;
}
{
sub.f16x2 r1500, r1435, r1436;
}
{
mul.f16x2 r1503, r1500, r1415;
}
{
sub.f16x2 r1506, r1441, r1442;
}
{
mul.f16x2 r1509, r1506, r1417;
}
{
add.f16x2 r1512, r1503, r1509;
}
{
add.f16x2 r1515, r1497, r1512;
}
{
add.f16x2 r1518, r1423, r1424;
}
{
mul.f16x2 r1521, r1518, r1416;
}
{
add.f16x2 r1524, r1426, r1521;
}
{
add.f16x2 r1527, r1429, r1430;
}
{
mul.f16x2 r1530, r1527, r1418;
}
{
add.f16x2 r1533, r1524, r1530;
}
{
sub.f16x2 r1536, r1435, r1436;
}
{
mul.f16x2 r1539, r1536, r1417;
}
{
sub.f16x2 r1542, r1441, r1442;
}
{
mul.f16x2 r1545, r1542, r1420;
}
{
add.f16x2 r1548, r1539, r1545;
}
{
sub.f16x2 r1551, r1533, r1548;
}
{
add.f16x2 r1554, r1423, r1424;
}
{
mul.f16x2 r1557, r1554, r1416;
}
{
add.f16x2 r1560, r1426, r1557;
}
{
add.f16x2 r1563, r1429, r1430;
}
{
mul.f16x2 r1566, r1563, r1418;
}
{
add.f16x2 r1569, r1560, r1566;
}
{
sub.f16x2 r1572, r1435, r1436;
}
{
mul.f16x2 r1575, r1572, r1417;
}
{
sub.f16x2 r1578, r1441, r1442;
}
{
mul.f16x2 r1581, r1578, r1420;
}
{
add.f16x2 r1584, r1575, r1581;
}
{
add.f16x2 r1587, r1569, r1584;
}
{
add.f16x2 r1590, r1435, r1436;
}
{
mul.f16x2 r1593, r1590, r1414;
}
{
add.f16x2 r1596, r1438, r1593;
}
{
add.f16x2 r1599, r1441, r1442;
}
{
mul.f16x2 r1602, r1599, r1416;
}
{
add.f16x2 r1605, r1596, r1602;
}
{
sub.f16x2 r1608, r1423, r1424;
}
{
mul.f16x2 r1611, r1608, r1415;
}
{
sub.f16x2 r1614, r1429, r1430;
}
{
mul.f16x2 r1617, r1614, r1417;
}
{
add.f16x2 r1620, r1611, r1617;
}
{
add.f16x2 r1623, r1605, r1620;
}
{
add.f16x2 r1626, r1435, r1436;
}
{
mul.f16x2 r1629, r1626, r1414;
}
{
add.f16x2 r1632, r1438, r1629;
}
{
add.f16x2 r1635, r1441, r1442;
}
{
mul.f16x2 r1638, r1635, r1416;
}
{
add.f16x2 r1641, r1632, r1638;
}
{
sub.f16x2 r1644, r1423, r1424;
}
{
mul.f16x2 r1647, r1644, r1415;
}
{
sub.f16x2 r1650, r1429, r1430;
}
{
mul.f16x2 r1653, r1650, r1417;
}
{
add.f16x2 r1656, r1647, r1653;
}
{
sub.f16x2 r1659, r1641, r1656;
}
{
add.f16x2 r1662, r1435, r1436;
}
{
mul.f16x2 r1665, r1662, r1416;
}
{
add.f16x2 r1668, r1438, r1665;
}
{
add.f16x2 r1671, r1441, r1442;
}
{
mul.f16x2 r1674, r1671, r1418;
}
{
add.f16x2 r1677, r1668, r1674;
}
{
sub.f16x2 r1680, r1423, r1424;
}
{
mul.f16x2 r1683, r1680, r1417;
}
{
sub.f16x2 r1686, r1429, r1430;
}
{
mul.f16x2 r1689, r1686, r1420;
}
{
add.f16x2 r1692, r1683, r1689;
}
{
add.f16x2 r1695, r1677, r1692;
}
{
add.f16x2 r1698, r1435, r1436;
}
{
mul.f16x2 r1701, r1698, r1416;
}
{
add.f16x2 r1704, r1438, r1701;
}
{
add.f16x2 r1707, r1441, r1442;
}
{
mul.f16x2 r1710, r1707, r1418;
}
{
add.f16x2 r1713, r1704, r1710;
}
{
sub.f16x2 r1716, r1423, r1424;
}
{
mul.f16x2 r1719, r1716, r1417;
}
{
sub.f16x2 r1722, r1429, r1430;
}
{
mul.f16x2 r1725, r1722, r1420;
}
{
add.f16x2 r1728, r1719, r1725;
}
{
sub.f16x2 r1731, r1713, r1728;
}
mul.wide.u32 rd8, r2211, 274877907;
shr.u64 rd9, rd8, 35;
cvt.u32.u64 r2228, rd9;
cvt.rn.f32.u32 f118, r2228;
mul.f32 f119, f118, 0f3E80ADFD;
cos.approx.f32 f85, f119;
sin.approx.f32 f120, f119;
neg.f32 f86, f120;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f85;
cvt.rn.f16.f32 high, f86;
mov.b32 r1734, {low, high};
}
mul.lo.s32 r2229, r2228, 125;
sub.s32 r2230, r2211, r2229;
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1737, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1739, {high, high};
}
{
mul.f16x2 r1741, r1623, r1739;
}
{
neg.f16x2 r1744, r1741;
}
{
fma.rn.f16x2 r1746, r1479, r1737, r1744;
}
{
mul.f16x2 r1750, r1479, r1739;
}
{
fma.rn.f16x2 r1753, r1623, r1737, r1750;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1757, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1759, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1761, {low, high};
}
{
mul.f16x2 r1762, r1759, r1761;
}
{
mul.f16x2 r1765, r1734, r1757;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1768, {high, low};
}
{
fma.rn.f16x2 r1770, r1762, r1768, r1765;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1774, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1776, {high, high};
}
{
mul.f16x2 r1778, r1695, r1776;
}
{
neg.f16x2 r1781, r1778;
}
{
fma.rn.f16x2 r1783, r1551, r1774, r1781;
}
{
mul.f16x2 r1787, r1551, r1776;
}
{
fma.rn.f16x2 r1790, r1695, r1774, r1787;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1794, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1796, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1798, {low, high};
}
{
mul.f16x2 r1799, r1796, r1798;
}
{
mul.f16x2 r1802, r1770, r1794;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1770;
mov.b32 r1805, {high, low};
}
{
fma.rn.f16x2 r1807, r1799, r1805, r1802;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1811, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1813, {high, high};
}
{
mul.f16x2 r1815, r1731, r1813;
}
{
neg.f16x2 r1818, r1815;
}
{
fma.rn.f16x2 r1820, r1587, r1811, r1818;
}
{
mul.f16x2 r1824, r1587, r1813;
}
{
fma.rn.f16x2 r1827, r1731, r1811, r1824;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1831, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1734;
mov.b32 r1833, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f93;
cvt.rn.f16.f32 high, f94;
mov.b32 r1835, {low, high};
}
{
mul.f16x2 r1836, r1833, r1835;
}
{
mul.f16x2 r1839, r1807, r1831;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1807;
mov.b32 r1842, {high, low};
}
{
fma.rn.f16x2 r1844, r1836, r1842, r1839;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1844;
mov.b32 r1848, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1844;
mov.b32 r1850, {high, high};
}
{
mul.f16x2 r1852, r1659, r1850;
}
{
neg.f16x2 r1855, r1852;
}
{
fma.rn.f16x2 r1857, r1515, r1848, r1855;
}
{
mul.f16x2 r1861, r1515, r1850;
}
{
fma.rn.f16x2 r1864, r1659, r1848, r1861;
}
shl.b32 r2231, r2230, 2;
add.s32 r2232, r2212, r2231;
barrier.sync 0;
mad.lo.s32 r2233, r2228, 2500, r2232;
st.shared.u32 [r2233], r1431;
st.shared.u32 [r2233+500], r1746;
st.shared.u32 [r2233+1000], r1783;
st.shared.u32 [r2233+1500], r1820;
st.shared.u32 [r2233+2000], r1857;
barrier.sync 0;
ld.shared.u32 r1897, [r2215];
ld.shared.u32 r1894, [r2215+2500];
ld.shared.u32 r1900, [r2215+5000];
ld.shared.u32 r1901, [r2215+7500];
ld.shared.u32 r1895, [r2215+10000];
barrier.sync 0;
st.shared.u32 [r2233], r1443;
st.shared.u32 [r2233+500], r1753;
st.shared.u32 [r2233+1000], r1790;
st.shared.u32 [r2233+1500], r1827;
st.shared.u32 [r2233+2000], r1864;
barrier.sync 0;
ld.shared.u32 r1909, [r2215];
ld.shared.u32 r1906, [r2215+2500];
ld.shared.u32 r1912, [r2215+5000];
ld.shared.u32 r1913, [r2215+7500];
ld.shared.u32 r1907, [r2215+10000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1885, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1886, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f102;
cvt.rn.f16.f32 high, f102;
mov.b32 r1887, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f104;
cvt.rn.f16.f32 high, f104;
mov.b32 r1888, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f106;
cvt.rn.f16.f32 high, f106;
mov.b32 r1889, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f108;
cvt.rn.f16.f32 high, f108;
mov.b32 r1890, {low, high};
}
{
neg.f16x2 r1891, r1890;
}
{
add.f16x2 r1893, r1894, r1895;
}
{
add.f16x2 r1896, r1897, r1893;
}
{
add.f16x2 r1899, r1900, r1901;
}
{
add.f16x2 %0, r1896, r1899;
}
{
add.f16x2 r1905, r1906, r1907;
}
{
add.f16x2 r1908, r1909, r1905;
}
{
add.f16x2 r1911, r1912, r1913;
}
{
add.f16x2 %1, r1908, r1911;
}
{
add.f16x2 r1917, r1894, r1895;
}
{
mul.f16x2 r1920, r1917, r1885;
}
{
add.f16x2 r1923, r1897, r1920;
}
{
add.f16x2 r1926, r1900, r1901;
}
{
mul.f16x2 r1929, r1926, r1887;
}
{
add.f16x2 r1932, r1923, r1929;
}
{
sub.f16x2 r1935, r1906, r1907;
}
{
mul.f16x2 r1938, r1935, r1886;
}
{
sub.f16x2 r1941, r1912, r1913;
}
{
mul.f16x2 r1944, r1941, r1888;
}
{
add.f16x2 r1947, r1938, r1944;
}
{
sub.f16x2 %2, r1932, r1947;
}
{
add.f16x2 r1953, r1894, r1895;
}
{
mul.f16x2 r1956, r1953, r1885;
}
{
add.f16x2 r1959, r1897, r1956;
}
{
add.f16x2 r1962, r1900, r1901;
}
{
mul.f16x2 r1965, r1962, r1887;
}
{
add.f16x2 r1968, r1959, r1965;
}
{
sub.f16x2 r1971, r1906, r1907;
}
{
mul.f16x2 r1974, r1971, r1886;
}
{
sub.f16x2 r1977, r1912, r1913;
}
{
mul.f16x2 r1980, r1977, r1888;
}
{
add.f16x2 r1983, r1974, r1980;
}
{
add.f16x2 %8, r1968, r1983;
}
{
add.f16x2 r1989, r1894, r1895;
}
{
mul.f16x2 r1992, r1989, r1887;
}
{
add.f16x2 r1995, r1897, r1992;
}
{
add.f16x2 r1998, r1900, r1901;
}
{
mul.f16x2 r2001, r1998, r1889;
}
{
add.f16x2 r2004, r1995, r2001;
}
{
sub.f16x2 r2007, r1906, r1907;
}
{
mul.f16x2 r2010, r2007, r1888;
}
{
sub.f16x2 r2013, r1912, r1913;
}
{
mul.f16x2 r2016, r2013, r1891;
}
{
add.f16x2 r2019, r2010, r2016;
}
{
sub.f16x2 %4, r2004, r2019;
}
{
add.f16x2 r2025, r1894, r1895;
}
{
mul.f16x2 r2028, r2025, r1887;
}
{
add.f16x2 r2031, r1897, r2028;
}
{
add.f16x2 r2034, r1900, r1901;
}
{
mul.f16x2 r2037, r2034, r1889;
}
{
add.f16x2 r2040, r2031, r2037;
}
{
sub.f16x2 r2043, r1906, r1907;
}
{
mul.f16x2 r2046, r2043, r1888;
}
{
sub.f16x2 r2049, r1912, r1913;
}
{
mul.f16x2 r2052, r2049, r1891;
}
{
add.f16x2 r2055, r2046, r2052;
}
{
add.f16x2 %6, r2040, r2055;
}
{
add.f16x2 r2061, r1906, r1907;
}
{
mul.f16x2 r2064, r2061, r1885;
}
{
add.f16x2 r2067, r1909, r2064;
}
{
add.f16x2 r2070, r1912, r1913;
}
{
mul.f16x2 r2073, r2070, r1887;
}
{
add.f16x2 r2076, r2067, r2073;
}
{
sub.f16x2 r2079, r1894, r1895;
}
{
mul.f16x2 r2082, r2079, r1886;
}
{
sub.f16x2 r2085, r1900, r1901;
}
{
mul.f16x2 r2088, r2085, r1888;
}
{
add.f16x2 r2091, r2082, r2088;
}
{
add.f16x2 %3, r2076, r2091;
}
{
add.f16x2 r2097, r1906, r1907;
}
{
mul.f16x2 r2100, r2097, r1885;
}
{
add.f16x2 r2103, r1909, r2100;
}
{
add.f16x2 r2106, r1912, r1913;
}
{
mul.f16x2 r2109, r2106, r1887;
}
{
add.f16x2 r2112, r2103, r2109;
}
{
sub.f16x2 r2115, r1894, r1895;
}
{
mul.f16x2 r2118, r2115, r1886;
}
{
sub.f16x2 r2121, r1900, r1901;
}
{
mul.f16x2 r2124, r2121, r1888;
}
{
add.f16x2 r2127, r2118, r2124;
}
{
sub.f16x2 %9, r2112, r2127;
}
{
add.f16x2 r2133, r1906, r1907;
}
{
mul.f16x2 r2136, r2133, r1887;
}
{
add.f16x2 r2139, r1909, r2136;
}
{
add.f16x2 r2142, r1912, r1913;
}
{
mul.f16x2 r2145, r2142, r1889;
}
{
add.f16x2 r2148, r2139, r2145;
}
{
sub.f16x2 r2151, r1894, r1895;
}
{
mul.f16x2 r2154, r2151, r1888;
}
{
sub.f16x2 r2157, r1900, r1901;
}
{
mul.f16x2 r2160, r2157, r1891;
}
{
add.f16x2 r2163, r2154, r2160;
}
{
add.f16x2 %5, r2148, r2163;
}
{
add.f16x2 r2169, r1906, r1907;
}
{
mul.f16x2 r2172, r2169, r1887;
}
{
add.f16x2 r2175, r1909, r2172;
}
{
add.f16x2 r2178, r1912, r1913;
}
{
mul.f16x2 r2181, r2178, r1889;
}
{
add.f16x2 r2184, r2175, r2181;
}
{
sub.f16x2 r2187, r1894, r1895;
}
{
mul.f16x2 r2190, r2187, r1888;
}
{
sub.f16x2 r2193, r1900, r1901;
}
{
mul.f16x2 r2196, r2193, r1891;
}
{
add.f16x2 r2199, r2190, r2196;
}
{
sub.f16x2 %7, r2184, r2199;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};


#endif
