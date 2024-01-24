#ifndef CUFFTDX_FFT_625_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_625_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<909, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<488>;
.reg .b32 r<7962>;
.reg .b64 rd<4>;
mov.u32 r7960, %tid.y;
mov.u32 r7961, %50;
mad.lo.s32 r7902, r7960, 5000, r7961;
mov.u32 r7903, %tid.x;
mov.f32 f482, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1, {low, high};
}
mov.f32 f484, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2, {low, high};
}
mov.f32 f478, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r3, {low, high};
}
mov.f32 f480, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 r15, %60, %94;
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
add.f16x2 r42, %60, %94;
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
add.f16x2 r78, %60, %94;
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
add.f16x2 r114, %60, %94;
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
add.f16x2 r150, %60, %94;
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
sub.f16x2 r201, %60, %94;
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
sub.f16x2 r237, %60, %94;
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
sub.f16x2 r273, %60, %94;
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
sub.f16x2 r309, %60, %94;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r644, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r961, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r962, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r965, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1281, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r1283, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r1284, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1285, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1286, {low, high};
}
{
neg.f16x2 r1287, r1286;
}
{
add.f16x2 r1289, %53, %90;
}
{
add.f16x2 r1292, %59, r1289;
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
add.f16x2 r1319, %59, r1316;
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
add.f16x2 r1355, %59, r1352;
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
add.f16x2 r1391, %59, r1388;
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
add.f16x2 r1427, %59, r1424;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1905, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1906, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r1907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r1908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1909, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2226, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2227, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2228, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2545, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2546, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2547, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2548, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2549, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2866, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2868, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2869, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r3185, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r3186, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r3187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r3188, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r3189, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
mul.wide.u32 rd2, r7903, 1374389535;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r7904, rd3;
mul.lo.s32 r7905, r7904, 25;
sub.s32 r7906, r7903, r7905;
cvt.rn.f32.u32 f485, r7906;
mul.f32 f486, f485, 0f3C24B5BE;
cos.approx.f32 f217, f486;
sin.approx.f32 f487, f486;
neg.f32 f218, f487;
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
mov.f32 f265, 0fBF800000;
mov.f32 f266, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
mad.lo.s32 r7907, r7904, 5000, r7902;
barrier.sync 0;
mad.lo.s32 r7908, r7906, 200, r7907;
st.shared.v2.f32 [r7908], {r1922, r1934};
st.shared.v2.f32 [r7908+8], {r3517, r3524};
st.shared.v2.f32 [r7908+16], {r3554, r3561};
st.shared.v2.f32 [r7908+24], {r3591, r3598};
st.shared.v2.f32 [r7908+32], {r3628, r3635};
st.shared.v2.f32 [r7908+40], {r3665, r3672};
st.shared.v2.f32 [r7908+48], {r3702, r3709};
st.shared.v2.f32 [r7908+56], {r3739, r3746};
st.shared.v2.f32 [r7908+64], {r3776, r3783};
st.shared.v2.f32 [r7908+72], {r3813, r3820};
st.shared.v2.f32 [r7908+80], {r3850, r3857};
st.shared.v2.f32 [r7908+88], {r3887, r3894};
st.shared.v2.f32 [r7908+96], {r3924, r3931};
st.shared.v2.f32 [r7908+104], {r3961, r3968};
st.shared.v2.f32 [r7908+112], {r3998, r4005};
st.shared.v2.f32 [r7908+120], {r4035, r4042};
st.shared.v2.f32 [r7908+128], {r4072, r4079};
st.shared.v2.f32 [r7908+136], {r4109, r4116};
st.shared.v2.f32 [r7908+144], {r4146, r4153};
st.shared.v2.f32 [r7908+152], {r4183, r4190};
st.shared.v2.f32 [r7908+160], {r4220, r4227};
st.shared.v2.f32 [r7908+168], {r4257, r4264};
st.shared.v2.f32 [r7908+176], {r4294, r4301};
st.shared.v2.f32 [r7908+184], {r4331, r4338};
st.shared.v2.f32 [r7908+192], {r4368, r4375};
barrier.sync 0;
mad.lo.s32 r7909, r7906, -192, r7908;
ld.shared.u32 r4408, [r7909];
ld.shared.u32 r4420, [r7909+4];
ld.shared.u32 r4728, [r7909+200];
ld.shared.u32 r4740, [r7909+204];
ld.shared.u32 r5048, [r7909+400];
ld.shared.u32 r5060, [r7909+404];
ld.shared.u32 r5368, [r7909+600];
ld.shared.u32 r5380, [r7909+604];
ld.shared.u32 r5688, [r7909+800];
ld.shared.u32 r5700, [r7909+804];
ld.shared.u32 r4405, [r7909+1000];
ld.shared.u32 r4417, [r7909+1004];
ld.shared.u32 r4725, [r7909+1200];
ld.shared.u32 r4737, [r7909+1204];
ld.shared.u32 r5045, [r7909+1400];
ld.shared.u32 r5057, [r7909+1404];
ld.shared.u32 r5365, [r7909+1600];
ld.shared.u32 r5377, [r7909+1604];
ld.shared.u32 r5685, [r7909+1800];
ld.shared.u32 r5697, [r7909+1804];
ld.shared.u32 r4411, [r7909+2000];
ld.shared.u32 r4423, [r7909+2004];
ld.shared.u32 r4731, [r7909+2200];
ld.shared.u32 r4743, [r7909+2204];
ld.shared.u32 r5051, [r7909+2400];
ld.shared.u32 r5063, [r7909+2404];
ld.shared.u32 r5371, [r7909+2600];
ld.shared.u32 r5383, [r7909+2604];
ld.shared.u32 r5691, [r7909+2800];
ld.shared.u32 r5703, [r7909+2804];
ld.shared.u32 r4412, [r7909+3000];
ld.shared.u32 r4424, [r7909+3004];
ld.shared.u32 r4732, [r7909+3200];
ld.shared.u32 r4744, [r7909+3204];
ld.shared.u32 r5052, [r7909+3400];
ld.shared.u32 r5064, [r7909+3404];
ld.shared.u32 r5372, [r7909+3600];
ld.shared.u32 r5384, [r7909+3604];
ld.shared.u32 r5692, [r7909+3800];
ld.shared.u32 r5704, [r7909+3804];
ld.shared.u32 r4406, [r7909+4000];
ld.shared.u32 r4418, [r7909+4004];
ld.shared.u32 r4726, [r7909+4200];
ld.shared.u32 r4738, [r7909+4204];
ld.shared.u32 r5046, [r7909+4400];
ld.shared.u32 r5058, [r7909+4404];
ld.shared.u32 r5366, [r7909+4600];
ld.shared.u32 r5378, [r7909+4604];
ld.shared.u32 r5686, [r7909+4800];
ld.shared.u32 r5698, [r7909+4804];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r4397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r4398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r4717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r4718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5038, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5039, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5680, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6302, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %0, r6311, r6314;
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
add.f16x2 %1, r6323, r6326;
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
sub.f16x2 %10, r6347, r6362;
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
add.f16x2 %40, r6383, r6398;
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
sub.f16x2 %20, r6419, r6434;
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
add.f16x2 %30, r6455, r6470;
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
add.f16x2 %11, r6491, r6506;
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
sub.f16x2 %41, r6527, r6542;
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
add.f16x2 %21, r6563, r6578;
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
sub.f16x2 %31, r6599, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6623, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %2, r6631, r6634;
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
add.f16x2 %3, r6643, r6646;
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
sub.f16x2 %12, r6667, r6682;
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
add.f16x2 %42, r6703, r6718;
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
sub.f16x2 %22, r6739, r6754;
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
add.f16x2 %32, r6775, r6790;
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
add.f16x2 %13, r6811, r6826;
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
sub.f16x2 %43, r6847, r6862;
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
add.f16x2 %23, r6883, r6898;
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
sub.f16x2 %33, r6919, r6934;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %4, r6951, r6954;
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
add.f16x2 %5, r6963, r6966;
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
sub.f16x2 %14, r6987, r7002;
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
add.f16x2 %44, r7023, r7038;
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
sub.f16x2 %24, r7059, r7074;
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
add.f16x2 %34, r7095, r7110;
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
add.f16x2 %15, r7131, r7146;
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
sub.f16x2 %45, r7167, r7182;
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
add.f16x2 %25, r7203, r7218;
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
sub.f16x2 %35, r7239, r7254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r7261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r7262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r7263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %6, r7271, r7274;
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
add.f16x2 %7, r7283, r7286;
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
sub.f16x2 %16, r7307, r7322;
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
add.f16x2 %46, r7343, r7358;
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
sub.f16x2 %26, r7379, r7394;
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
add.f16x2 %36, r7415, r7430;
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
add.f16x2 %17, r7451, r7466;
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
sub.f16x2 %47, r7487, r7502;
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
add.f16x2 %27, r7523, r7538;
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
sub.f16x2 %37, r7559, r7574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r7581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r7582, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r7583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7584, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %8, r7591, r7594;
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
add.f16x2 %9, r7603, r7606;
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
sub.f16x2 %18, r7627, r7642;
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
add.f16x2 %48, r7663, r7678;
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
sub.f16x2 %28, r7699, r7714;
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
add.f16x2 %38, r7735, r7750;
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
add.f16x2 %19, r7771, r7786;
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
sub.f16x2 %49, r7807, r7822;
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
add.f16x2 %29, r7843, r7858;
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
sub.f16x2 %39, r7879, r7894;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[18].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<908, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<488>;
.reg .b32 r<7962>;
.reg .b64 rd<4>;
mov.u32 r7960, %tid.y;
mov.u32 r7961, %50;
mad.lo.s32 r7902, r7960, 2500, r7961;
mov.u32 r7903, %tid.x;
mov.f32 f482, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1, {low, high};
}
mov.f32 f484, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2, {low, high};
}
mov.f32 f478, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r3, {low, high};
}
mov.f32 f480, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 r15, %60, %94;
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
add.f16x2 r42, %60, %94;
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
add.f16x2 r78, %60, %94;
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
add.f16x2 r114, %60, %94;
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
add.f16x2 r150, %60, %94;
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
sub.f16x2 r201, %60, %94;
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
sub.f16x2 r237, %60, %94;
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
sub.f16x2 r273, %60, %94;
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
sub.f16x2 r309, %60, %94;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r322, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r323, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r324, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r325, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r641, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r642, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r643, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r644, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r645, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r961, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r962, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r963, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r964, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r965, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1281, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1282, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r1283, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r1284, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1285, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1286, {low, high};
}
{
neg.f16x2 r1287, r1286;
}
{
add.f16x2 r1289, %53, %90;
}
{
add.f16x2 r1292, %59, r1289;
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
add.f16x2 r1319, %59, r1316;
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
add.f16x2 r1355, %59, r1352;
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
add.f16x2 r1391, %59, r1388;
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
add.f16x2 r1427, %59, r1424;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1905, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r1906, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r1907, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r1908, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r1909, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2225, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2226, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2227, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2228, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2229, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2545, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2546, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2547, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2548, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2549, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2865, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r2866, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r2867, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r2868, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r2869, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r3185, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r3186, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r3187, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r3188, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r3189, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
mul.wide.u32 rd2, r7903, 1374389535;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r7904, rd3;
mul.lo.s32 r7905, r7904, 25;
sub.s32 r7906, r7903, r7905;
mad.lo.s32 r7907, r7904, 2500, r7902;
cvt.rn.f32.u32 f485, r7906;
mul.f32 f486, f485, 0f3C24B5BE;
cos.approx.f32 f217, f486;
sin.approx.f32 f487, f486;
neg.f32 f218, f487;
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
mov.f32 f265, 0fBF800000;
mov.f32 f266, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
cvt.rn.f16.f32 low, f265;
cvt.rn.f16.f32 high, f266;
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
mad.lo.s32 r7908, r7906, 100, r7907;
st.shared.u32 [r7908], r1922;
st.shared.u32 [r7908+4], r3517;
st.shared.u32 [r7908+8], r3554;
st.shared.u32 [r7908+12], r3591;
st.shared.u32 [r7908+16], r3628;
st.shared.u32 [r7908+20], r3665;
st.shared.u32 [r7908+24], r3702;
st.shared.u32 [r7908+28], r3739;
st.shared.u32 [r7908+32], r3776;
st.shared.u32 [r7908+36], r3813;
st.shared.u32 [r7908+40], r3850;
st.shared.u32 [r7908+44], r3887;
st.shared.u32 [r7908+48], r3924;
st.shared.u32 [r7908+52], r3961;
st.shared.u32 [r7908+56], r3998;
st.shared.u32 [r7908+60], r4035;
st.shared.u32 [r7908+64], r4072;
st.shared.u32 [r7908+68], r4109;
st.shared.u32 [r7908+72], r4146;
st.shared.u32 [r7908+76], r4183;
st.shared.u32 [r7908+80], r4220;
st.shared.u32 [r7908+84], r4257;
st.shared.u32 [r7908+88], r4294;
st.shared.u32 [r7908+92], r4331;
st.shared.u32 [r7908+96], r4368;
barrier.sync 0;
mad.lo.s32 r7909, r7906, -96, r7908;
ld.shared.u32 r4408, [r7909];
ld.shared.u32 r4728, [r7909+100];
ld.shared.u32 r5048, [r7909+200];
ld.shared.u32 r5368, [r7909+300];
ld.shared.u32 r5688, [r7909+400];
ld.shared.u32 r4405, [r7909+500];
ld.shared.u32 r4725, [r7909+600];
ld.shared.u32 r5045, [r7909+700];
ld.shared.u32 r5365, [r7909+800];
ld.shared.u32 r5685, [r7909+900];
ld.shared.u32 r4411, [r7909+1000];
ld.shared.u32 r4731, [r7909+1100];
ld.shared.u32 r5051, [r7909+1200];
ld.shared.u32 r5371, [r7909+1300];
ld.shared.u32 r5691, [r7909+1400];
ld.shared.u32 r4412, [r7909+1500];
ld.shared.u32 r4732, [r7909+1600];
ld.shared.u32 r5052, [r7909+1700];
ld.shared.u32 r5372, [r7909+1800];
ld.shared.u32 r5692, [r7909+1900];
ld.shared.u32 r4406, [r7909+2000];
ld.shared.u32 r4726, [r7909+2100];
ld.shared.u32 r5046, [r7909+2200];
ld.shared.u32 r5366, [r7909+2300];
ld.shared.u32 r5686, [r7909+2400];
barrier.sync 0;
st.shared.u32 [r7908], r1934;
st.shared.u32 [r7908+4], r3524;
st.shared.u32 [r7908+8], r3561;
st.shared.u32 [r7908+12], r3598;
st.shared.u32 [r7908+16], r3635;
st.shared.u32 [r7908+20], r3672;
st.shared.u32 [r7908+24], r3709;
st.shared.u32 [r7908+28], r3746;
st.shared.u32 [r7908+32], r3783;
st.shared.u32 [r7908+36], r3820;
st.shared.u32 [r7908+40], r3857;
st.shared.u32 [r7908+44], r3894;
st.shared.u32 [r7908+48], r3931;
st.shared.u32 [r7908+52], r3968;
st.shared.u32 [r7908+56], r4005;
st.shared.u32 [r7908+60], r4042;
st.shared.u32 [r7908+64], r4079;
st.shared.u32 [r7908+68], r4116;
st.shared.u32 [r7908+72], r4153;
st.shared.u32 [r7908+76], r4190;
st.shared.u32 [r7908+80], r4227;
st.shared.u32 [r7908+84], r4264;
st.shared.u32 [r7908+88], r4301;
st.shared.u32 [r7908+92], r4338;
st.shared.u32 [r7908+96], r4375;
barrier.sync 0;
ld.shared.u32 r4420, [r7909];
ld.shared.u32 r4740, [r7909+100];
ld.shared.u32 r5060, [r7909+200];
ld.shared.u32 r5380, [r7909+300];
ld.shared.u32 r5700, [r7909+400];
ld.shared.u32 r4417, [r7909+500];
ld.shared.u32 r4737, [r7909+600];
ld.shared.u32 r5057, [r7909+700];
ld.shared.u32 r5377, [r7909+800];
ld.shared.u32 r5697, [r7909+900];
ld.shared.u32 r4423, [r7909+1000];
ld.shared.u32 r4743, [r7909+1100];
ld.shared.u32 r5063, [r7909+1200];
ld.shared.u32 r5383, [r7909+1300];
ld.shared.u32 r5703, [r7909+1400];
ld.shared.u32 r4424, [r7909+1500];
ld.shared.u32 r4744, [r7909+1600];
ld.shared.u32 r5064, [r7909+1700];
ld.shared.u32 r5384, [r7909+1800];
ld.shared.u32 r5704, [r7909+1900];
ld.shared.u32 r4418, [r7909+2000];
ld.shared.u32 r4738, [r7909+2100];
ld.shared.u32 r5058, [r7909+2200];
ld.shared.u32 r5378, [r7909+2300];
ld.shared.u32 r5698, [r7909+2400];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r4397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r4398, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4399, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4400, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4716, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r4717, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r4718, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r4719, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r4720, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5036, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5037, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5038, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5039, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5040, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5356, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5357, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5358, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5359, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5360, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5676, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r5677, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r5678, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r5679, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r5680, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6300, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6302, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6303, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6304, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %0, r6311, r6314;
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
add.f16x2 %1, r6323, r6326;
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
sub.f16x2 %10, r6347, r6362;
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
add.f16x2 %40, r6383, r6398;
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
sub.f16x2 %20, r6419, r6434;
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
add.f16x2 %30, r6455, r6470;
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
add.f16x2 %11, r6491, r6506;
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
sub.f16x2 %41, r6527, r6542;
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
add.f16x2 %21, r6563, r6578;
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
sub.f16x2 %31, r6599, r6614;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6620, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6621, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6622, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6623, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6624, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %2, r6631, r6634;
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
add.f16x2 %3, r6643, r6646;
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
sub.f16x2 %12, r6667, r6682;
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
add.f16x2 %42, r6703, r6718;
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
sub.f16x2 %22, r6739, r6754;
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
add.f16x2 %32, r6775, r6790;
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
add.f16x2 %13, r6811, r6826;
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
sub.f16x2 %43, r6847, r6862;
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
add.f16x2 %23, r6883, r6898;
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
sub.f16x2 %33, r6919, r6934;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6940, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r6941, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r6942, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r6943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r6944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %4, r6951, r6954;
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
add.f16x2 %5, r6963, r6966;
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
sub.f16x2 %14, r6987, r7002;
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
add.f16x2 %44, r7023, r7038;
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
sub.f16x2 %24, r7059, r7074;
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
add.f16x2 %34, r7095, r7110;
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
add.f16x2 %15, r7131, r7146;
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
sub.f16x2 %45, r7167, r7182;
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
add.f16x2 %25, r7203, r7218;
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
sub.f16x2 %35, r7239, r7254;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7260, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r7261, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r7262, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r7263, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7264, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %6, r7271, r7274;
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
add.f16x2 %7, r7283, r7286;
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
sub.f16x2 %16, r7307, r7322;
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
add.f16x2 %46, r7343, r7358;
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
sub.f16x2 %26, r7379, r7394;
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
add.f16x2 %36, r7415, r7430;
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
add.f16x2 %17, r7451, r7466;
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
sub.f16x2 %47, r7487, r7502;
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
add.f16x2 %27, r7523, r7538;
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
sub.f16x2 %37, r7559, r7574;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7580, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
mov.b32 r7581, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f478;
cvt.rn.f16.f32 high, f478;
mov.b32 r7582, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f480;
cvt.rn.f16.f32 high, f480;
mov.b32 r7583, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f482;
cvt.rn.f16.f32 high, f482;
mov.b32 r7584, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f484;
cvt.rn.f16.f32 high, f484;
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
add.f16x2 %8, r7591, r7594;
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
add.f16x2 %9, r7603, r7606;
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
sub.f16x2 %18, r7627, r7642;
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
add.f16x2 %48, r7663, r7678;
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
sub.f16x2 %28, r7699, r7714;
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
add.f16x2 %38, r7735, r7750;
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
add.f16x2 %19, r7771, r7786;
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
sub.f16x2 %49, r7807, r7822;
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
add.f16x2 %29, r7843, r7858;
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
sub.f16x2 %39, r7879, r7894;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[18].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<910, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<94>;
.reg .b32 r<1757>;
.reg .b64 rd<8>;
mov.u32 r1734, %tid.x;
mov.f32 f82, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1, {low, high};
}
mov.f32 f84, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r2, {low, high};
}
mov.f32 f78, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r3, {low, high};
}
mov.f32 f80, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd2, r1734, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1735, rd3;
mul.lo.s32 r1736, r1735, 125;
sub.s32 r1737, r1734, r1736;
cvt.rn.f32.u32 f85, r1737;
mul.f32 f86, f85, 0f3C24B5BE;
cos.approx.f32 f13, f86;
sin.approx.f32 f87, f86;
neg.f32 f14, f87;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r321, {low, high};
}
mov.u32 r1738, %tid.y;
mov.u32 r1739, %10;
mad.lo.s32 r1740, r1738, 5000, r1739;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
mad.lo.s32 r1741, r1735, 5000, r1740;
barrier.sync 0;
mad.lo.s32 r1742, r1737, 40, r1741;
st.shared.v2.f32 [r1742], {r18, r30};
st.shared.v2.f32 [r1742+8], {r333, r340};
st.shared.v2.f32 [r1742+16], {r370, r377};
st.shared.v2.f32 [r1742+24], {r407, r414};
st.shared.v2.f32 [r1742+32], {r444, r451};
barrier.sync 0;
shl.b32 r1743, r1737, 5;
sub.s32 r1744, r1742, r1743;
ld.shared.u32 r484, [r1744];
ld.shared.u32 r496, [r1744+4];
ld.shared.u32 r481, [r1744+1000];
ld.shared.u32 r493, [r1744+1004];
ld.shared.u32 r487, [r1744+2000];
ld.shared.u32 r499, [r1744+2004];
ld.shared.u32 r488, [r1744+3000];
ld.shared.u32 r500, [r1744+3004];
ld.shared.u32 r482, [r1744+4000];
ld.shared.u32 r494, [r1744+4004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd4, r1737, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1745, rd5;
cvt.rn.f32.u32 f88, r1745;
mul.f32 f89, f88, 0f3D4DE32E;
cos.approx.f32 f37, f89;
sin.approx.f32 f90, f89;
neg.f32 f38, f90;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r792, {low, high};
}
mul.lo.s32 r1746, r1745, 5;
sub.s32 r1747, r1737, r1746;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
shl.b32 r1748, r1747, 3;
add.s32 r1749, r1741, r1748;
barrier.sync 0;
mad.lo.s32 r1750, r1745, 200, r1749;
st.shared.u32 [r1750], r489;
st.shared.u32 [r1750+4], r501;
st.shared.u32 [r1750+40], r804;
st.shared.u32 [r1750+44], r811;
st.shared.u32 [r1750+80], r841;
st.shared.u32 [r1750+84], r848;
st.shared.u32 [r1750+120], r878;
st.shared.u32 [r1750+124], r885;
st.shared.u32 [r1750+160], r915;
st.shared.u32 [r1750+164], r922;
barrier.sync 0;
ld.shared.u32 r955, [r1744];
ld.shared.u32 r967, [r1744+4];
ld.shared.u32 r952, [r1744+1000];
ld.shared.u32 r964, [r1744+1004];
ld.shared.u32 r958, [r1744+2000];
ld.shared.u32 r970, [r1744+2004];
ld.shared.u32 r959, [r1744+3000];
ld.shared.u32 r971, [r1744+3004];
ld.shared.u32 r953, [r1744+4000];
ld.shared.u32 r965, [r1744+4004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r946, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd6, r1737, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1751, rd7;
cvt.rn.f32.u32 f91, r1751;
mul.f32 f92, f91, 0f3E80ADFD;
cos.approx.f32 f61, f92;
sin.approx.f32 f93, f92;
neg.f32 f62, f93;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1263, {low, high};
}
mul.lo.s32 r1752, r1751, 25;
sub.s32 r1753, r1737, r1752;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
shl.b32 r1754, r1753, 3;
add.s32 r1755, r1741, r1754;
barrier.sync 0;
mad.lo.s32 r1756, r1751, 1000, r1755;
st.shared.u32 [r1756], r960;
st.shared.u32 [r1756+4], r972;
st.shared.u32 [r1756+200], r1275;
st.shared.u32 [r1756+204], r1282;
st.shared.u32 [r1756+400], r1312;
st.shared.u32 [r1756+404], r1319;
st.shared.u32 [r1756+600], r1349;
st.shared.u32 [r1756+604], r1356;
st.shared.u32 [r1756+800], r1386;
st.shared.u32 [r1756+804], r1393;
barrier.sync 0;
ld.shared.u32 r1426, [r1744];
ld.shared.u32 r1438, [r1744+4];
ld.shared.u32 r1423, [r1744+1000];
ld.shared.u32 r1435, [r1744+1004];
ld.shared.u32 r1429, [r1744+2000];
ld.shared.u32 r1441, [r1744+2004];
ld.shared.u32 r1430, [r1744+3000];
ld.shared.u32 r1442, [r1744+3004];
ld.shared.u32 r1424, [r1744+4000];
ld.shared.u32 r1436, [r1744+4004];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1414, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r1416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r1417, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
add.f16x2 %0, r1425, r1428;
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
add.f16x2 %1, r1437, r1440;
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
sub.f16x2 %2, r1461, r1476;
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
add.f16x2 %8, r1497, r1512;
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
sub.f16x2 %4, r1533, r1548;
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
add.f16x2 %6, r1569, r1584;
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
add.f16x2 %3, r1605, r1620;
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
sub.f16x2 %9, r1641, r1656;
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
add.f16x2 %5, r1677, r1692;
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
sub.f16x2 %7, r1713, r1728;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<911, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<94>;
.reg .b32 r<1757>;
.reg .b64 rd<8>;
mov.u32 r1734, %tid.y;
mov.u32 r1735, %10;
mad.lo.s32 r1736, r1734, 2500, r1735;
mov.u32 r1737, %tid.x;
mov.f32 f82, 0f3E9E377A;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1, {low, high};
}
mov.f32 f84, 0fBF737871;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r2, {low, high};
}
mov.f32 f78, 0fBF4F1BBD;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r3, {low, high};
}
mov.f32 f80, 0fBF167918;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r4, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r5, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd2, r1737, 274877907;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1738, rd3;
mul.lo.s32 r1739, r1738, 125;
sub.s32 r1740, r1737, r1739;
mad.lo.s32 r1741, r1738, 2500, r1736;
cvt.rn.f32.u32 f85, r1740;
mul.f32 f86, f85, 0f3C24B5BE;
cos.approx.f32 f13, f86;
sin.approx.f32 f87, f86;
neg.f32 f14, f87;
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
mov.f32 f69, 0fBF800000;
mov.f32 f70, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
barrier.sync 0;
mad.lo.s32 r1742, r1740, 20, r1741;
st.shared.u32 [r1742], r18;
st.shared.u32 [r1742+4], r333;
st.shared.u32 [r1742+8], r370;
st.shared.u32 [r1742+12], r407;
st.shared.u32 [r1742+16], r444;
barrier.sync 0;
shl.b32 r1743, r1740, 4;
sub.s32 r1744, r1742, r1743;
ld.shared.u32 r484, [r1744];
ld.shared.u32 r481, [r1744+500];
ld.shared.u32 r487, [r1744+1000];
ld.shared.u32 r488, [r1744+1500];
ld.shared.u32 r482, [r1744+2000];
barrier.sync 0;
st.shared.u32 [r1742], r30;
st.shared.u32 [r1742+4], r340;
st.shared.u32 [r1742+8], r377;
st.shared.u32 [r1742+12], r414;
st.shared.u32 [r1742+16], r451;
barrier.sync 0;
ld.shared.u32 r496, [r1744];
ld.shared.u32 r493, [r1744+500];
ld.shared.u32 r499, [r1744+1000];
ld.shared.u32 r500, [r1744+1500];
ld.shared.u32 r494, [r1744+2000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r472, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r473, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r474, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r475, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r476, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd4, r1740, -858993459;
shr.u64 rd5, rd4, 34;
cvt.u32.u64 r1745, rd5;
mul.lo.s32 r1746, r1745, 5;
sub.s32 r1747, r1740, r1746;
shl.b32 r1748, r1747, 2;
add.s32 r1749, r1741, r1748;
cvt.rn.f32.u32 f88, r1745;
mul.f32 f89, f88, 0f3D4DE32E;
cos.approx.f32 f37, f89;
sin.approx.f32 f90, f89;
neg.f32 f38, f90;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f37;
cvt.rn.f16.f32 high, f38;
mov.b32 r792, {low, high};
}
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
barrier.sync 0;
mad.lo.s32 r1750, r1745, 100, r1749;
st.shared.u32 [r1750], r489;
st.shared.u32 [r1750+20], r804;
st.shared.u32 [r1750+40], r841;
st.shared.u32 [r1750+60], r878;
st.shared.u32 [r1750+80], r915;
barrier.sync 0;
ld.shared.u32 r955, [r1744];
ld.shared.u32 r952, [r1744+500];
ld.shared.u32 r958, [r1744+1000];
ld.shared.u32 r959, [r1744+1500];
ld.shared.u32 r953, [r1744+2000];
barrier.sync 0;
st.shared.u32 [r1750], r501;
st.shared.u32 [r1750+20], r811;
st.shared.u32 [r1750+40], r848;
st.shared.u32 [r1750+60], r885;
st.shared.u32 [r1750+80], r922;
barrier.sync 0;
ld.shared.u32 r967, [r1744];
ld.shared.u32 r964, [r1744+500];
ld.shared.u32 r970, [r1744+1000];
ld.shared.u32 r971, [r1744+1500];
ld.shared.u32 r965, [r1744+2000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r943, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r944, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r945, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r946, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r947, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
mul.wide.u32 rd6, r1740, 1374389535;
shr.u64 rd7, rd6, 35;
cvt.u32.u64 r1751, rd7;
mul.lo.s32 r1752, r1751, 25;
sub.s32 r1753, r1740, r1752;
shl.b32 r1754, r1753, 2;
add.s32 r1755, r1741, r1754;
cvt.rn.f32.u32 f91, r1751;
mul.f32 f92, f91, 0f3E80ADFD;
cos.approx.f32 f61, f92;
sin.approx.f32 f93, f92;
neg.f32 f62, f93;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f61;
cvt.rn.f16.f32 high, f62;
mov.b32 r1263, {low, high};
}
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
cvt.rn.f16.f32 low, f69;
cvt.rn.f16.f32 high, f70;
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
barrier.sync 0;
mad.lo.s32 r1756, r1751, 500, r1755;
st.shared.u32 [r1756], r960;
st.shared.u32 [r1756+100], r1275;
st.shared.u32 [r1756+200], r1312;
st.shared.u32 [r1756+300], r1349;
st.shared.u32 [r1756+400], r1386;
barrier.sync 0;
ld.shared.u32 r1426, [r1744];
ld.shared.u32 r1423, [r1744+500];
ld.shared.u32 r1429, [r1744+1000];
ld.shared.u32 r1430, [r1744+1500];
ld.shared.u32 r1424, [r1744+2000];
barrier.sync 0;
st.shared.u32 [r1756], r972;
st.shared.u32 [r1756+100], r1282;
st.shared.u32 [r1756+200], r1319;
st.shared.u32 [r1756+300], r1356;
st.shared.u32 [r1756+400], r1393;
barrier.sync 0;
ld.shared.u32 r1438, [r1744];
ld.shared.u32 r1435, [r1744+500];
ld.shared.u32 r1441, [r1744+1000];
ld.shared.u32 r1442, [r1744+1500];
ld.shared.u32 r1436, [r1744+2000];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1414, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r1415, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r1416, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r1417, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r1418, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
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
add.f16x2 %0, r1425, r1428;
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
add.f16x2 %1, r1437, r1440;
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
sub.f16x2 %2, r1461, r1476;
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
add.f16x2 %8, r1497, r1512;
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
sub.f16x2 %4, r1533, r1548;
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
add.f16x2 %6, r1569, r1584;
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
add.f16x2 %3, r1605, r1620;
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
sub.f16x2 %9, r1641, r1656;
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
add.f16x2 %5, r1677, r1692;
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
sub.f16x2 %7, r1713, r1728;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)));
};


#endif
