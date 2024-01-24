#ifndef CUFFTDX_FFT_144_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_144_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<1146, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<230>;
.reg .b32 r<1744>;
.reg .b64 rd<5>;
mov.u32 r1731, %tid.y;
shl.b32 r1732, r1731, 1;
mov.u32 r1733, %24;
mad.lo.s32 r1734, r1732, 576, r1733;
mov.u32 r1735, %tid.x;
mov.f32 f196, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %33, %41;
}
{
add.f16x2 r6, %25, r3;
}
{
add.f16x2 r9, %34, %42;
}
{
add.f16x2 r12, %26, r9;
}
{
add.f16x2 r15, %33, %41;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %25, r18;
}
{
sub.f16x2 r24, %34, %42;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %33, %41;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %25, r36;
}
{
sub.f16x2 r42, %34, %42;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %34, %42;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %26, r54;
}
{
sub.f16x2 r60, %33, %41;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %34, %42;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %26, r72;
}
{
sub.f16x2 r78, %33, %41;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %37, %45;
}
{
add.f16x2 r92, %29, r89;
}
{
add.f16x2 r95, %38, %46;
}
{
add.f16x2 r98, %30, r95;
}
{
add.f16x2 r101, %37, %45;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %29, r104;
}
{
sub.f16x2 r110, %38, %46;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %37, %45;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %29, r122;
}
{
sub.f16x2 r128, %38, %46;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %38, %46;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %30, r140;
}
{
sub.f16x2 r146, %37, %45;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %38, %46;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %30, r158;
}
{
sub.f16x2 r164, %37, %45;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
mov.f32 f202, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r173, {low, high};
}
mov.f32 f198, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r176, {low, high};
}
mov.f32 f123, 0fBF800000;
{
mul.f16x2 r183, r116, r173;
}
{
mul.f16x2 r186, r152, r174;
}
{
sub.f16x2 r189, r183, r186;
}
{
mul.f16x2 r192, r116, r174;
}
{
fma.rn.f16x2 r195, r152, r173, r192;
}
{
mul.f16x2 r199, r134, r175;
}
{
mul.f16x2 r202, r170, r176;
}
{
sub.f16x2 r205, r199, r202;
}
{
mul.f16x2 r208, r134, r176;
}
{
fma.rn.f16x2 r211, r170, r175, r208;
}
{
add.f16x2 r215, r6, r92;
}
{
add.f16x2 r218, r12, r98;
}
{
sub.f16x2 r221, r6, r92;
}
{
sub.f16x2 r224, r12, r98;
}
{
add.f16x2 r227, r30, r189;
}
{
add.f16x2 r230, r66, r195;
}
{
sub.f16x2 r233, r30, r189;
}
{
sub.f16x2 r236, r66, r195;
}
{
add.f16x2 r239, r48, r205;
}
{
add.f16x2 r242, r84, r211;
}
{
sub.f16x2 r245, r48, r205;
}
{
sub.f16x2 r248, r84, r211;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r252, {low, high};
}
{
add.f16x2 r253, %35, %43;
}
{
add.f16x2 r256, %27, r253;
}
{
add.f16x2 r259, %36, %44;
}
{
add.f16x2 r262, %28, r259;
}
{
add.f16x2 r265, %35, %43;
}
{
mul.f16x2 r268, r265, r251;
}
{
add.f16x2 r271, %27, r268;
}
{
sub.f16x2 r274, %36, %44;
}
{
mul.f16x2 r277, r274, r252;
}
{
add.f16x2 r280, r271, r277;
}
{
add.f16x2 r283, %35, %43;
}
{
mul.f16x2 r286, r283, r251;
}
{
add.f16x2 r289, %27, r286;
}
{
sub.f16x2 r292, %36, %44;
}
{
mul.f16x2 r295, r292, r252;
}
{
sub.f16x2 r298, r289, r295;
}
{
add.f16x2 r301, %36, %44;
}
{
mul.f16x2 r304, r301, r251;
}
{
add.f16x2 r307, %28, r304;
}
{
sub.f16x2 r310, %35, %43;
}
{
mul.f16x2 r313, r310, r252;
}
{
sub.f16x2 r316, r307, r313;
}
{
add.f16x2 r319, %36, %44;
}
{
mul.f16x2 r322, r319, r251;
}
{
add.f16x2 r325, %28, r322;
}
{
sub.f16x2 r328, %35, %43;
}
{
mul.f16x2 r331, r328, r252;
}
{
add.f16x2 r334, r325, r331;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r337, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r338, {low, high};
}
{
add.f16x2 r339, %39, %47;
}
{
add.f16x2 r342, %31, r339;
}
{
add.f16x2 r345, %40, %48;
}
{
add.f16x2 r348, %32, r345;
}
{
add.f16x2 r351, %39, %47;
}
{
mul.f16x2 r354, r351, r337;
}
{
add.f16x2 r357, %31, r354;
}
{
sub.f16x2 r360, %40, %48;
}
{
mul.f16x2 r363, r360, r338;
}
{
add.f16x2 r366, r357, r363;
}
{
add.f16x2 r369, %39, %47;
}
{
mul.f16x2 r372, r369, r337;
}
{
add.f16x2 r375, %31, r372;
}
{
sub.f16x2 r378, %40, %48;
}
{
mul.f16x2 r381, r378, r338;
}
{
sub.f16x2 r384, r375, r381;
}
{
add.f16x2 r387, %40, %48;
}
{
mul.f16x2 r390, r387, r337;
}
{
add.f16x2 r393, %32, r390;
}
{
sub.f16x2 r396, %39, %47;
}
{
mul.f16x2 r399, r396, r338;
}
{
sub.f16x2 r402, r393, r399;
}
{
add.f16x2 r405, %40, %48;
}
{
mul.f16x2 r408, r405, r337;
}
{
add.f16x2 r411, %32, r408;
}
{
sub.f16x2 r414, %39, %47;
}
{
mul.f16x2 r417, r414, r338;
}
{
add.f16x2 r420, r411, r417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r423, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r426, {low, high};
}
{
mul.f16x2 r433, r366, r423;
}
{
mul.f16x2 r436, r402, r424;
}
{
sub.f16x2 r439, r433, r436;
}
{
mul.f16x2 r442, r366, r424;
}
{
fma.rn.f16x2 r445, r402, r423, r442;
}
{
mul.f16x2 r449, r384, r425;
}
{
mul.f16x2 r452, r420, r426;
}
{
sub.f16x2 r455, r449, r452;
}
{
mul.f16x2 r458, r384, r426;
}
{
fma.rn.f16x2 r461, r420, r425, r458;
}
{
add.f16x2 r465, r256, r342;
}
{
add.f16x2 r468, r262, r348;
}
{
sub.f16x2 r471, r256, r342;
}
{
sub.f16x2 r474, r262, r348;
}
{
add.f16x2 r477, r280, r439;
}
{
add.f16x2 r480, r316, r445;
}
{
sub.f16x2 r483, r280, r439;
}
{
sub.f16x2 r486, r316, r445;
}
{
add.f16x2 r489, r298, r455;
}
{
add.f16x2 r492, r334, r461;
}
{
sub.f16x2 r495, r298, r455;
}
{
sub.f16x2 r498, r334, r461;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r501, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r502, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r503, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r504, {low, high};
}
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r507, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r508, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r510, {low, high};
}
{
mul.f16x2 r523, r477, r501;
}
{
mul.f16x2 r526, r480, r502;
}
{
sub.f16x2 r529, r523, r526;
}
{
mul.f16x2 r532, r477, r502;
}
{
fma.rn.f16x2 r535, r480, r501, r532;
}
{
mul.f16x2 r539, r489, r503;
}
{
mul.f16x2 r542, r492, r504;
}
{
sub.f16x2 r545, r539, r542;
}
{
mul.f16x2 r548, r489, r504;
}
{
fma.rn.f16x2 r551, r492, r503, r548;
}
{
neg.f16x2 r555, r474;
}
{
mul.f16x2 r557, r483, r507;
}
{
mul.f16x2 r560, r486, r508;
}
{
sub.f16x2 r563, r557, r560;
}
{
mul.f16x2 r566, r483, r508;
}
{
fma.rn.f16x2 r569, r486, r507, r566;
}
{
mul.f16x2 r573, r495, r509;
}
{
mul.f16x2 r576, r498, r510;
}
{
sub.f16x2 r579, r573, r576;
}
{
mul.f16x2 r582, r495, r510;
}
{
fma.rn.f16x2 r585, r498, r509, r582;
}
{
add.f16x2 r589, r215, r465;
}
{
add.f16x2 r592, r218, r468;
}
{
sub.f16x2 r595, r215, r465;
}
{
sub.f16x2 r598, r218, r468;
}
{
add.f16x2 r601, r227, r529;
}
{
add.f16x2 r604, r230, r535;
}
{
sub.f16x2 r607, r227, r529;
}
{
sub.f16x2 r610, r230, r535;
}
{
add.f16x2 r613, r239, r545;
}
{
add.f16x2 r616, r242, r551;
}
{
sub.f16x2 r619, r239, r545;
}
{
sub.f16x2 r622, r242, r551;
}
{
add.f16x2 r625, r221, r555;
}
{
add.f16x2 r628, r224, r471;
}
{
sub.f16x2 r631, r221, r555;
}
{
sub.f16x2 r634, r224, r471;
}
{
add.f16x2 r637, r233, r563;
}
{
add.f16x2 r640, r236, r569;
}
{
sub.f16x2 r643, r233, r563;
}
{
sub.f16x2 r646, r236, r569;
}
{
add.f16x2 r649, r245, r579;
}
{
add.f16x2 r652, r248, r585;
}
{
sub.f16x2 r655, r245, r579;
}
{
sub.f16x2 r658, r248, r585;
}
mul.wide.u32 rd2, r1735, -1431655765;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1736, rd3;
mul.lo.s32 r1737, r1736, 12;
sub.s32 r1738, r1735, r1737;
shr.u64 rd4, rd2, 34;
cvt.u32.u64 r1739, rd4;
and.b32 r1740, r1739, 1073741822;
mad.lo.s32 r1741, r1740, 576, r1734;
cvt.rn.f32.u32 f227, r1738;
mul.f32 f228, f227, 0f3D32B8C2;
cos.approx.f32 f101, f228;
sin.approx.f32 f229, f228;
neg.f32 f102, f229;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f101;
cvt.rn.f16.f32 high, f102;
mov.b32 r661, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r664, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r666, {high, high};
}
{
mul.f16x2 r668, r604, r666;
}
{
fma.rn.f16x2 r671, r601, r664, r668;
}
{
mul.f16x2 r675, r601, r666;
}
{
neg.f16x2 r678, r675;
}
{
fma.rn.f16x2 r680, r604, r664, r678;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r684, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r686, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r688, {low, high};
}
{
mul.f16x2 r689, r686, r688;
}
{
mul.f16x2 r692, r661, r684;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r695, {high, low};
}
{
fma.rn.f16x2 r697, r689, r695, r692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r701, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r703, {high, high};
}
{
mul.f16x2 r705, r616, r703;
}
{
fma.rn.f16x2 r708, r613, r701, r705;
}
{
mul.f16x2 r712, r613, r703;
}
{
neg.f16x2 r715, r712;
}
{
fma.rn.f16x2 r717, r616, r701, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r721, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r723, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r725, {low, high};
}
{
mul.f16x2 r726, r723, r725;
}
{
mul.f16x2 r729, r697, r721;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r732, {high, low};
}
{
fma.rn.f16x2 r734, r726, r732, r729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r738, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r740, {high, high};
}
{
mul.f16x2 r742, r628, r740;
}
{
fma.rn.f16x2 r745, r625, r738, r742;
}
{
mul.f16x2 r749, r625, r740;
}
{
neg.f16x2 r752, r749;
}
{
fma.rn.f16x2 r754, r628, r738, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r758, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r760, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r762, {low, high};
}
{
mul.f16x2 r763, r760, r762;
}
{
mul.f16x2 r766, r734, r758;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r769, {high, low};
}
{
fma.rn.f16x2 r771, r763, r769, r766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r775, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r777, {high, high};
}
{
mul.f16x2 r779, r640, r777;
}
{
fma.rn.f16x2 r782, r637, r775, r779;
}
{
mul.f16x2 r786, r637, r777;
}
{
neg.f16x2 r789, r786;
}
{
fma.rn.f16x2 r791, r640, r775, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r795, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r797, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r799, {low, high};
}
{
mul.f16x2 r800, r797, r799;
}
{
mul.f16x2 r803, r771, r795;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r806, {high, low};
}
{
fma.rn.f16x2 r808, r800, r806, r803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r812, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r814, {high, high};
}
{
mul.f16x2 r816, r652, r814;
}
{
fma.rn.f16x2 r819, r649, r812, r816;
}
{
mul.f16x2 r823, r649, r814;
}
{
neg.f16x2 r826, r823;
}
{
fma.rn.f16x2 r828, r652, r812, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r832, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r834, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r836, {low, high};
}
{
mul.f16x2 r837, r834, r836;
}
{
mul.f16x2 r840, r808, r832;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r843, {high, low};
}
{
fma.rn.f16x2 r845, r837, r843, r840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r849, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r851, {high, high};
}
{
mul.f16x2 r853, r598, r851;
}
{
fma.rn.f16x2 r856, r595, r849, r853;
}
{
mul.f16x2 r860, r595, r851;
}
{
neg.f16x2 r863, r860;
}
{
fma.rn.f16x2 r865, r598, r849, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r869, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r871, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r873, {low, high};
}
{
mul.f16x2 r874, r871, r873;
}
{
mul.f16x2 r877, r845, r869;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r880, {high, low};
}
{
fma.rn.f16x2 r882, r874, r880, r877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r886, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r888, {high, high};
}
{
mul.f16x2 r890, r610, r888;
}
{
fma.rn.f16x2 r893, r607, r886, r890;
}
{
mul.f16x2 r897, r607, r888;
}
{
neg.f16x2 r900, r897;
}
{
fma.rn.f16x2 r902, r610, r886, r900;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r906, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r908, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r910, {low, high};
}
{
mul.f16x2 r911, r908, r910;
}
{
mul.f16x2 r914, r882, r906;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r917, {high, low};
}
{
fma.rn.f16x2 r919, r911, r917, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r923, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r925, {high, high};
}
{
mul.f16x2 r927, r622, r925;
}
{
fma.rn.f16x2 r930, r619, r923, r927;
}
{
mul.f16x2 r934, r619, r925;
}
{
neg.f16x2 r937, r934;
}
{
fma.rn.f16x2 r939, r622, r923, r937;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r943, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r945, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r947, {low, high};
}
{
mul.f16x2 r948, r945, r947;
}
{
mul.f16x2 r951, r919, r943;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r954, {high, low};
}
{
fma.rn.f16x2 r956, r948, r954, r951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r960, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r962, {high, high};
}
{
mul.f16x2 r964, r634, r962;
}
{
fma.rn.f16x2 r967, r631, r960, r964;
}
{
mul.f16x2 r971, r631, r962;
}
{
neg.f16x2 r974, r971;
}
{
fma.rn.f16x2 r976, r634, r960, r974;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r980, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r982, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r984, {low, high};
}
{
mul.f16x2 r985, r982, r984;
}
{
mul.f16x2 r988, r956, r980;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r991, {high, low};
}
{
fma.rn.f16x2 r993, r985, r991, r988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r997, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r999, {high, high};
}
{
mul.f16x2 r1001, r646, r999;
}
{
fma.rn.f16x2 r1004, r643, r997, r1001;
}
{
mul.f16x2 r1008, r643, r999;
}
{
neg.f16x2 r1011, r1008;
}
{
fma.rn.f16x2 r1013, r646, r997, r1011;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r1017, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r1019, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1021, {low, high};
}
{
mul.f16x2 r1022, r1019, r1021;
}
{
mul.f16x2 r1025, r993, r1017;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r1028, {high, low};
}
{
fma.rn.f16x2 r1030, r1022, r1028, r1025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1030;
mov.b32 r1034, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1030;
mov.b32 r1036, {high, high};
}
{
mul.f16x2 r1038, r658, r1036;
}
{
fma.rn.f16x2 r1041, r655, r1034, r1038;
}
{
mul.f16x2 r1045, r655, r1036;
}
{
neg.f16x2 r1048, r1045;
}
{
fma.rn.f16x2 r1050, r658, r1034, r1048;
}
barrier.sync 0;
mad.lo.s32 r1742, r1738, 96, r1741;
st.shared.v4.f32 [r1742], {r589, r592, r671, r680};
st.shared.v4.f32 [r1742+16], {r708, r717, r745, r754};
st.shared.v4.f32 [r1742+32], {r782, r791, r819, r828};
st.shared.v4.f32 [r1742+48], {r856, r865, r893, r902};
st.shared.v4.f32 [r1742+64], {r930, r939, r967, r976};
st.shared.v4.f32 [r1742+80], {r1004, r1013, r1041, r1050};
barrier.sync 0;
mad.lo.s32 r1743, r1738, -88, r1742;
ld.shared.u32 r1077, [r1743];
ld.shared.u32 r1083, [r1743+4];
ld.shared.u32 r1327, [r1743+96];
ld.shared.u32 r1333, [r1743+100];
ld.shared.u32 r1163, [r1743+192];
ld.shared.u32 r1169, [r1743+196];
ld.shared.u32 r1413, [r1743+288];
ld.shared.u32 r1419, [r1743+292];
ld.shared.u32 r1074, [r1743+384];
ld.shared.u32 r1080, [r1743+388];
ld.shared.u32 r1324, [r1743+480];
ld.shared.u32 r1330, [r1743+484];
ld.shared.u32 r1160, [r1743+576];
ld.shared.u32 r1166, [r1743+580];
ld.shared.u32 r1410, [r1743+672];
ld.shared.u32 r1416, [r1743+676];
ld.shared.u32 r1075, [r1743+768];
ld.shared.u32 r1081, [r1743+772];
ld.shared.u32 r1325, [r1743+864];
ld.shared.u32 r1331, [r1743+868];
ld.shared.u32 r1161, [r1743+960];
ld.shared.u32 r1167, [r1743+964];
ld.shared.u32 r1411, [r1743+1056];
ld.shared.u32 r1417, [r1743+1060];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1071, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1072, {low, high};
}
{
add.f16x2 r1073, r1074, r1075;
}
{
add.f16x2 r1076, r1077, r1073;
}
{
add.f16x2 r1079, r1080, r1081;
}
{
add.f16x2 r1082, r1083, r1079;
}
{
add.f16x2 r1085, r1074, r1075;
}
{
mul.f16x2 r1088, r1085, r1071;
}
{
add.f16x2 r1091, r1077, r1088;
}
{
sub.f16x2 r1094, r1080, r1081;
}
{
mul.f16x2 r1097, r1094, r1072;
}
{
add.f16x2 r1100, r1091, r1097;
}
{
add.f16x2 r1103, r1074, r1075;
}
{
mul.f16x2 r1106, r1103, r1071;
}
{
add.f16x2 r1109, r1077, r1106;
}
{
sub.f16x2 r1112, r1080, r1081;
}
{
mul.f16x2 r1115, r1112, r1072;
}
{
sub.f16x2 r1118, r1109, r1115;
}
{
add.f16x2 r1121, r1080, r1081;
}
{
mul.f16x2 r1124, r1121, r1071;
}
{
add.f16x2 r1127, r1083, r1124;
}
{
sub.f16x2 r1130, r1074, r1075;
}
{
mul.f16x2 r1133, r1130, r1072;
}
{
sub.f16x2 r1136, r1127, r1133;
}
{
add.f16x2 r1139, r1080, r1081;
}
{
mul.f16x2 r1142, r1139, r1071;
}
{
add.f16x2 r1145, r1083, r1142;
}
{
sub.f16x2 r1148, r1074, r1075;
}
{
mul.f16x2 r1151, r1148, r1072;
}
{
add.f16x2 r1154, r1145, r1151;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1158, {low, high};
}
{
add.f16x2 r1159, r1160, r1161;
}
{
add.f16x2 r1162, r1163, r1159;
}
{
add.f16x2 r1165, r1166, r1167;
}
{
add.f16x2 r1168, r1169, r1165;
}
{
add.f16x2 r1171, r1160, r1161;
}
{
mul.f16x2 r1174, r1171, r1157;
}
{
add.f16x2 r1177, r1163, r1174;
}
{
sub.f16x2 r1180, r1166, r1167;
}
{
mul.f16x2 r1183, r1180, r1158;
}
{
add.f16x2 r1186, r1177, r1183;
}
{
add.f16x2 r1189, r1160, r1161;
}
{
mul.f16x2 r1192, r1189, r1157;
}
{
add.f16x2 r1195, r1163, r1192;
}
{
sub.f16x2 r1198, r1166, r1167;
}
{
mul.f16x2 r1201, r1198, r1158;
}
{
sub.f16x2 r1204, r1195, r1201;
}
{
add.f16x2 r1207, r1166, r1167;
}
{
mul.f16x2 r1210, r1207, r1157;
}
{
add.f16x2 r1213, r1169, r1210;
}
{
sub.f16x2 r1216, r1160, r1161;
}
{
mul.f16x2 r1219, r1216, r1158;
}
{
sub.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, r1166, r1167;
}
{
mul.f16x2 r1228, r1225, r1157;
}
{
add.f16x2 r1231, r1169, r1228;
}
{
sub.f16x2 r1234, r1160, r1161;
}
{
mul.f16x2 r1237, r1234, r1158;
}
{
add.f16x2 r1240, r1231, r1237;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1246, {low, high};
}
{
mul.f16x2 r1253, r1186, r1243;
}
{
mul.f16x2 r1256, r1222, r1244;
}
{
sub.f16x2 r1259, r1253, r1256;
}
{
mul.f16x2 r1262, r1186, r1244;
}
{
fma.rn.f16x2 r1265, r1222, r1243, r1262;
}
{
mul.f16x2 r1269, r1204, r1245;
}
{
mul.f16x2 r1272, r1240, r1246;
}
{
sub.f16x2 r1275, r1269, r1272;
}
{
mul.f16x2 r1278, r1204, r1246;
}
{
fma.rn.f16x2 r1281, r1240, r1245, r1278;
}
{
add.f16x2 r1285, r1076, r1162;
}
{
add.f16x2 r1288, r1082, r1168;
}
{
sub.f16x2 r1291, r1076, r1162;
}
{
sub.f16x2 r1294, r1082, r1168;
}
{
add.f16x2 r1297, r1100, r1259;
}
{
add.f16x2 r1300, r1136, r1265;
}
{
sub.f16x2 r1303, r1100, r1259;
}
{
sub.f16x2 r1306, r1136, r1265;
}
{
add.f16x2 r1309, r1118, r1275;
}
{
add.f16x2 r1312, r1154, r1281;
}
{
sub.f16x2 r1315, r1118, r1275;
}
{
sub.f16x2 r1318, r1154, r1281;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1322, {low, high};
}
{
add.f16x2 r1323, r1324, r1325;
}
{
add.f16x2 r1326, r1327, r1323;
}
{
add.f16x2 r1329, r1330, r1331;
}
{
add.f16x2 r1332, r1333, r1329;
}
{
add.f16x2 r1335, r1324, r1325;
}
{
mul.f16x2 r1338, r1335, r1321;
}
{
add.f16x2 r1341, r1327, r1338;
}
{
sub.f16x2 r1344, r1330, r1331;
}
{
mul.f16x2 r1347, r1344, r1322;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, r1324, r1325;
}
{
mul.f16x2 r1356, r1353, r1321;
}
{
add.f16x2 r1359, r1327, r1356;
}
{
sub.f16x2 r1362, r1330, r1331;
}
{
mul.f16x2 r1365, r1362, r1322;
}
{
sub.f16x2 r1368, r1359, r1365;
}
{
add.f16x2 r1371, r1330, r1331;
}
{
mul.f16x2 r1374, r1371, r1321;
}
{
add.f16x2 r1377, r1333, r1374;
}
{
sub.f16x2 r1380, r1324, r1325;
}
{
mul.f16x2 r1383, r1380, r1322;
}
{
sub.f16x2 r1386, r1377, r1383;
}
{
add.f16x2 r1389, r1330, r1331;
}
{
mul.f16x2 r1392, r1389, r1321;
}
{
add.f16x2 r1395, r1333, r1392;
}
{
sub.f16x2 r1398, r1324, r1325;
}
{
mul.f16x2 r1401, r1398, r1322;
}
{
add.f16x2 r1404, r1395, r1401;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1407, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1408, {low, high};
}
{
add.f16x2 r1409, r1410, r1411;
}
{
add.f16x2 r1412, r1413, r1409;
}
{
add.f16x2 r1415, r1416, r1417;
}
{
add.f16x2 r1418, r1419, r1415;
}
{
add.f16x2 r1421, r1410, r1411;
}
{
mul.f16x2 r1424, r1421, r1407;
}
{
add.f16x2 r1427, r1413, r1424;
}
{
sub.f16x2 r1430, r1416, r1417;
}
{
mul.f16x2 r1433, r1430, r1408;
}
{
add.f16x2 r1436, r1427, r1433;
}
{
add.f16x2 r1439, r1410, r1411;
}
{
mul.f16x2 r1442, r1439, r1407;
}
{
add.f16x2 r1445, r1413, r1442;
}
{
sub.f16x2 r1448, r1416, r1417;
}
{
mul.f16x2 r1451, r1448, r1408;
}
{
sub.f16x2 r1454, r1445, r1451;
}
{
add.f16x2 r1457, r1416, r1417;
}
{
mul.f16x2 r1460, r1457, r1407;
}
{
add.f16x2 r1463, r1419, r1460;
}
{
sub.f16x2 r1466, r1410, r1411;
}
{
mul.f16x2 r1469, r1466, r1408;
}
{
sub.f16x2 r1472, r1463, r1469;
}
{
add.f16x2 r1475, r1416, r1417;
}
{
mul.f16x2 r1478, r1475, r1407;
}
{
add.f16x2 r1481, r1419, r1478;
}
{
sub.f16x2 r1484, r1410, r1411;
}
{
mul.f16x2 r1487, r1484, r1408;
}
{
add.f16x2 r1490, r1481, r1487;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1493, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1494, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1495, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1496, {low, high};
}
{
mul.f16x2 r1503, r1436, r1493;
}
{
mul.f16x2 r1506, r1472, r1494;
}
{
sub.f16x2 r1509, r1503, r1506;
}
{
mul.f16x2 r1512, r1436, r1494;
}
{
fma.rn.f16x2 r1515, r1472, r1493, r1512;
}
{
mul.f16x2 r1519, r1454, r1495;
}
{
mul.f16x2 r1522, r1490, r1496;
}
{
sub.f16x2 r1525, r1519, r1522;
}
{
mul.f16x2 r1528, r1454, r1496;
}
{
fma.rn.f16x2 r1531, r1490, r1495, r1528;
}
{
add.f16x2 r1535, r1326, r1412;
}
{
add.f16x2 r1538, r1332, r1418;
}
{
sub.f16x2 r1541, r1326, r1412;
}
{
sub.f16x2 r1544, r1332, r1418;
}
{
add.f16x2 r1547, r1350, r1509;
}
{
add.f16x2 r1550, r1386, r1515;
}
{
sub.f16x2 r1553, r1350, r1509;
}
{
sub.f16x2 r1556, r1386, r1515;
}
{
add.f16x2 r1559, r1368, r1525;
}
{
add.f16x2 r1562, r1404, r1531;
}
{
sub.f16x2 r1565, r1368, r1525;
}
{
sub.f16x2 r1568, r1404, r1531;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1574, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1593, r1547, r1571;
}
{
mul.f16x2 r1596, r1550, r1572;
}
{
sub.f16x2 r1599, r1593, r1596;
}
{
mul.f16x2 r1602, r1547, r1572;
}
{
fma.rn.f16x2 r1605, r1550, r1571, r1602;
}
{
mul.f16x2 r1609, r1559, r1573;
}
{
mul.f16x2 r1612, r1562, r1574;
}
{
sub.f16x2 r1615, r1609, r1612;
}
{
mul.f16x2 r1618, r1559, r1574;
}
{
fma.rn.f16x2 r1621, r1562, r1573, r1618;
}
{
neg.f16x2 r1625, r1544;
}
{
mul.f16x2 r1627, r1553, r1577;
}
{
mul.f16x2 r1630, r1556, r1578;
}
{
sub.f16x2 r1633, r1627, r1630;
}
{
mul.f16x2 r1636, r1553, r1578;
}
{
fma.rn.f16x2 r1639, r1556, r1577, r1636;
}
{
mul.f16x2 r1643, r1565, r1579;
}
{
mul.f16x2 r1646, r1568, r1580;
}
{
sub.f16x2 r1649, r1643, r1646;
}
{
mul.f16x2 r1652, r1565, r1580;
}
{
fma.rn.f16x2 r1655, r1568, r1579, r1652;
}
{
add.f16x2 %0, r1285, r1535;
}
{
add.f16x2 %1, r1288, r1538;
}
{
sub.f16x2 %12, r1285, r1535;
}
{
sub.f16x2 %13, r1288, r1538;
}
{
add.f16x2 %2, r1297, r1599;
}
{
add.f16x2 %3, r1300, r1605;
}
{
sub.f16x2 %14, r1297, r1599;
}
{
sub.f16x2 %15, r1300, r1605;
}
{
add.f16x2 %4, r1309, r1615;
}
{
add.f16x2 %5, r1312, r1621;
}
{
sub.f16x2 %16, r1309, r1615;
}
{
sub.f16x2 %17, r1312, r1621;
}
{
add.f16x2 %6, r1291, r1625;
}
{
add.f16x2 %7, r1294, r1541;
}
{
sub.f16x2 %18, r1291, r1625;
}
{
sub.f16x2 %19, r1294, r1541;
}
{
add.f16x2 %8, r1303, r1633;
}
{
add.f16x2 %9, r1306, r1639;
}
{
sub.f16x2 %20, r1303, r1633;
}
{
sub.f16x2 %21, r1306, r1639;
}
{
add.f16x2 %10, r1315, r1649;
}
{
add.f16x2 %11, r1318, r1655;
}
{
sub.f16x2 %22, r1315, r1649;
}
{
sub.f16x2 %23, r1318, r1655;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<1147, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<230>;
.reg .b32 r<1741>;
.reg .b64 rd<4>;
mov.u32 r1731, %tid.y;
mov.u32 r1732, %24;
mad.lo.s32 r1733, r1731, 576, r1732;
mov.u32 r1734, %tid.x;
mov.f32 f196, 0fBF000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1, {low, high};
}
mov.f32 f200, 0fBF5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r2, {low, high};
}
{
add.f16x2 r3, %33, %41;
}
{
add.f16x2 r6, %25, r3;
}
{
add.f16x2 r9, %34, %42;
}
{
add.f16x2 r12, %26, r9;
}
{
add.f16x2 r15, %33, %41;
}
{
mul.f16x2 r18, r15, r1;
}
{
add.f16x2 r21, %25, r18;
}
{
sub.f16x2 r24, %34, %42;
}
{
mul.f16x2 r27, r24, r2;
}
{
add.f16x2 r30, r21, r27;
}
{
add.f16x2 r33, %33, %41;
}
{
mul.f16x2 r36, r33, r1;
}
{
add.f16x2 r39, %25, r36;
}
{
sub.f16x2 r42, %34, %42;
}
{
mul.f16x2 r45, r42, r2;
}
{
sub.f16x2 r48, r39, r45;
}
{
add.f16x2 r51, %34, %42;
}
{
mul.f16x2 r54, r51, r1;
}
{
add.f16x2 r57, %26, r54;
}
{
sub.f16x2 r60, %33, %41;
}
{
mul.f16x2 r63, r60, r2;
}
{
sub.f16x2 r66, r57, r63;
}
{
add.f16x2 r69, %34, %42;
}
{
mul.f16x2 r72, r69, r1;
}
{
add.f16x2 r75, %26, r72;
}
{
sub.f16x2 r78, %33, %41;
}
{
mul.f16x2 r81, r78, r2;
}
{
add.f16x2 r84, r75, r81;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r87, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r88, {low, high};
}
{
add.f16x2 r89, %37, %45;
}
{
add.f16x2 r92, %29, r89;
}
{
add.f16x2 r95, %38, %46;
}
{
add.f16x2 r98, %30, r95;
}
{
add.f16x2 r101, %37, %45;
}
{
mul.f16x2 r104, r101, r87;
}
{
add.f16x2 r107, %29, r104;
}
{
sub.f16x2 r110, %38, %46;
}
{
mul.f16x2 r113, r110, r88;
}
{
add.f16x2 r116, r107, r113;
}
{
add.f16x2 r119, %37, %45;
}
{
mul.f16x2 r122, r119, r87;
}
{
add.f16x2 r125, %29, r122;
}
{
sub.f16x2 r128, %38, %46;
}
{
mul.f16x2 r131, r128, r88;
}
{
sub.f16x2 r134, r125, r131;
}
{
add.f16x2 r137, %38, %46;
}
{
mul.f16x2 r140, r137, r87;
}
{
add.f16x2 r143, %30, r140;
}
{
sub.f16x2 r146, %37, %45;
}
{
mul.f16x2 r149, r146, r88;
}
{
sub.f16x2 r152, r143, r149;
}
{
add.f16x2 r155, %38, %46;
}
{
mul.f16x2 r158, r155, r87;
}
{
add.f16x2 r161, %30, r158;
}
{
sub.f16x2 r164, %37, %45;
}
{
mul.f16x2 r167, r164, r88;
}
{
add.f16x2 r170, r161, r167;
}
mov.f32 f202, 0f3F000000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r173, {low, high};
}
mov.f32 f198, 0f3F5DB3D7;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r174, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r175, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r176, {low, high};
}
mov.f32 f123, 0fBF800000;
{
mul.f16x2 r183, r116, r173;
}
{
mul.f16x2 r186, r152, r174;
}
{
sub.f16x2 r189, r183, r186;
}
{
mul.f16x2 r192, r116, r174;
}
{
fma.rn.f16x2 r195, r152, r173, r192;
}
{
mul.f16x2 r199, r134, r175;
}
{
mul.f16x2 r202, r170, r176;
}
{
sub.f16x2 r205, r199, r202;
}
{
mul.f16x2 r208, r134, r176;
}
{
fma.rn.f16x2 r211, r170, r175, r208;
}
{
add.f16x2 r215, r6, r92;
}
{
add.f16x2 r218, r12, r98;
}
{
sub.f16x2 r221, r6, r92;
}
{
sub.f16x2 r224, r12, r98;
}
{
add.f16x2 r227, r30, r189;
}
{
add.f16x2 r230, r66, r195;
}
{
sub.f16x2 r233, r30, r189;
}
{
sub.f16x2 r236, r66, r195;
}
{
add.f16x2 r239, r48, r205;
}
{
add.f16x2 r242, r84, r211;
}
{
sub.f16x2 r245, r48, r205;
}
{
sub.f16x2 r248, r84, r211;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r251, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r252, {low, high};
}
{
add.f16x2 r253, %35, %43;
}
{
add.f16x2 r256, %27, r253;
}
{
add.f16x2 r259, %36, %44;
}
{
add.f16x2 r262, %28, r259;
}
{
add.f16x2 r265, %35, %43;
}
{
mul.f16x2 r268, r265, r251;
}
{
add.f16x2 r271, %27, r268;
}
{
sub.f16x2 r274, %36, %44;
}
{
mul.f16x2 r277, r274, r252;
}
{
add.f16x2 r280, r271, r277;
}
{
add.f16x2 r283, %35, %43;
}
{
mul.f16x2 r286, r283, r251;
}
{
add.f16x2 r289, %27, r286;
}
{
sub.f16x2 r292, %36, %44;
}
{
mul.f16x2 r295, r292, r252;
}
{
sub.f16x2 r298, r289, r295;
}
{
add.f16x2 r301, %36, %44;
}
{
mul.f16x2 r304, r301, r251;
}
{
add.f16x2 r307, %28, r304;
}
{
sub.f16x2 r310, %35, %43;
}
{
mul.f16x2 r313, r310, r252;
}
{
sub.f16x2 r316, r307, r313;
}
{
add.f16x2 r319, %36, %44;
}
{
mul.f16x2 r322, r319, r251;
}
{
add.f16x2 r325, %28, r322;
}
{
sub.f16x2 r328, %35, %43;
}
{
mul.f16x2 r331, r328, r252;
}
{
add.f16x2 r334, r325, r331;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r337, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r338, {low, high};
}
{
add.f16x2 r339, %39, %47;
}
{
add.f16x2 r342, %31, r339;
}
{
add.f16x2 r345, %40, %48;
}
{
add.f16x2 r348, %32, r345;
}
{
add.f16x2 r351, %39, %47;
}
{
mul.f16x2 r354, r351, r337;
}
{
add.f16x2 r357, %31, r354;
}
{
sub.f16x2 r360, %40, %48;
}
{
mul.f16x2 r363, r360, r338;
}
{
add.f16x2 r366, r357, r363;
}
{
add.f16x2 r369, %39, %47;
}
{
mul.f16x2 r372, r369, r337;
}
{
add.f16x2 r375, %31, r372;
}
{
sub.f16x2 r378, %40, %48;
}
{
mul.f16x2 r381, r378, r338;
}
{
sub.f16x2 r384, r375, r381;
}
{
add.f16x2 r387, %40, %48;
}
{
mul.f16x2 r390, r387, r337;
}
{
add.f16x2 r393, %32, r390;
}
{
sub.f16x2 r396, %39, %47;
}
{
mul.f16x2 r399, r396, r338;
}
{
sub.f16x2 r402, r393, r399;
}
{
add.f16x2 r405, %40, %48;
}
{
mul.f16x2 r408, r405, r337;
}
{
add.f16x2 r411, %32, r408;
}
{
sub.f16x2 r414, %39, %47;
}
{
mul.f16x2 r417, r414, r338;
}
{
add.f16x2 r420, r411, r417;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r423, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r424, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r425, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r426, {low, high};
}
{
mul.f16x2 r433, r366, r423;
}
{
mul.f16x2 r436, r402, r424;
}
{
sub.f16x2 r439, r433, r436;
}
{
mul.f16x2 r442, r366, r424;
}
{
fma.rn.f16x2 r445, r402, r423, r442;
}
{
mul.f16x2 r449, r384, r425;
}
{
mul.f16x2 r452, r420, r426;
}
{
sub.f16x2 r455, r449, r452;
}
{
mul.f16x2 r458, r384, r426;
}
{
fma.rn.f16x2 r461, r420, r425, r458;
}
{
add.f16x2 r465, r256, r342;
}
{
add.f16x2 r468, r262, r348;
}
{
sub.f16x2 r471, r256, r342;
}
{
sub.f16x2 r474, r262, r348;
}
{
add.f16x2 r477, r280, r439;
}
{
add.f16x2 r480, r316, r445;
}
{
sub.f16x2 r483, r280, r439;
}
{
sub.f16x2 r486, r316, r445;
}
{
add.f16x2 r489, r298, r455;
}
{
add.f16x2 r492, r334, r461;
}
{
sub.f16x2 r495, r298, r455;
}
{
sub.f16x2 r498, r334, r461;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r501, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r502, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r503, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r504, {low, high};
}
mov.f32 f124, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r507, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r508, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r509, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r510, {low, high};
}
{
mul.f16x2 r523, r477, r501;
}
{
mul.f16x2 r526, r480, r502;
}
{
sub.f16x2 r529, r523, r526;
}
{
mul.f16x2 r532, r477, r502;
}
{
fma.rn.f16x2 r535, r480, r501, r532;
}
{
mul.f16x2 r539, r489, r503;
}
{
mul.f16x2 r542, r492, r504;
}
{
sub.f16x2 r545, r539, r542;
}
{
mul.f16x2 r548, r489, r504;
}
{
fma.rn.f16x2 r551, r492, r503, r548;
}
{
neg.f16x2 r555, r474;
}
{
mul.f16x2 r557, r483, r507;
}
{
mul.f16x2 r560, r486, r508;
}
{
sub.f16x2 r563, r557, r560;
}
{
mul.f16x2 r566, r483, r508;
}
{
fma.rn.f16x2 r569, r486, r507, r566;
}
{
mul.f16x2 r573, r495, r509;
}
{
mul.f16x2 r576, r498, r510;
}
{
sub.f16x2 r579, r573, r576;
}
{
mul.f16x2 r582, r495, r510;
}
{
fma.rn.f16x2 r585, r498, r509, r582;
}
{
add.f16x2 r589, r215, r465;
}
{
add.f16x2 r592, r218, r468;
}
{
sub.f16x2 r595, r215, r465;
}
{
sub.f16x2 r598, r218, r468;
}
{
add.f16x2 r601, r227, r529;
}
{
add.f16x2 r604, r230, r535;
}
{
sub.f16x2 r607, r227, r529;
}
{
sub.f16x2 r610, r230, r535;
}
{
add.f16x2 r613, r239, r545;
}
{
add.f16x2 r616, r242, r551;
}
{
sub.f16x2 r619, r239, r545;
}
{
sub.f16x2 r622, r242, r551;
}
{
add.f16x2 r625, r221, r555;
}
{
add.f16x2 r628, r224, r471;
}
{
sub.f16x2 r631, r221, r555;
}
{
sub.f16x2 r634, r224, r471;
}
{
add.f16x2 r637, r233, r563;
}
{
add.f16x2 r640, r236, r569;
}
{
sub.f16x2 r643, r233, r563;
}
{
sub.f16x2 r646, r236, r569;
}
{
add.f16x2 r649, r245, r579;
}
{
add.f16x2 r652, r248, r585;
}
{
sub.f16x2 r655, r245, r579;
}
{
sub.f16x2 r658, r248, r585;
}
mul.wide.u32 rd2, r1734, -1431655765;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r1735, rd3;
mul.lo.s32 r1736, r1735, 12;
sub.s32 r1737, r1734, r1736;
mad.lo.s32 r1738, r1735, 576, r1733;
cvt.rn.f32.u32 f227, r1737;
mul.f32 f228, f227, 0f3D32B8C2;
cos.approx.f32 f101, f228;
sin.approx.f32 f229, f228;
neg.f32 f102, f229;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f101;
cvt.rn.f16.f32 high, f102;
mov.b32 r661, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r664, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r666, {high, high};
}
{
mul.f16x2 r668, r604, r666;
}
{
fma.rn.f16x2 r671, r601, r664, r668;
}
{
mul.f16x2 r675, r601, r666;
}
{
neg.f16x2 r678, r675;
}
{
fma.rn.f16x2 r680, r604, r664, r678;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r684, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r686, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r688, {low, high};
}
{
mul.f16x2 r689, r686, r688;
}
{
mul.f16x2 r692, r661, r684;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r695, {high, low};
}
{
fma.rn.f16x2 r697, r689, r695, r692;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r701, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r703, {high, high};
}
{
mul.f16x2 r705, r616, r703;
}
{
fma.rn.f16x2 r708, r613, r701, r705;
}
{
mul.f16x2 r712, r613, r703;
}
{
neg.f16x2 r715, r712;
}
{
fma.rn.f16x2 r717, r616, r701, r715;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r721, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r723, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r725, {low, high};
}
{
mul.f16x2 r726, r723, r725;
}
{
mul.f16x2 r729, r697, r721;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r697;
mov.b32 r732, {high, low};
}
{
fma.rn.f16x2 r734, r726, r732, r729;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r738, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r740, {high, high};
}
{
mul.f16x2 r742, r628, r740;
}
{
fma.rn.f16x2 r745, r625, r738, r742;
}
{
mul.f16x2 r749, r625, r740;
}
{
neg.f16x2 r752, r749;
}
{
fma.rn.f16x2 r754, r628, r738, r752;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r758, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r760, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r762, {low, high};
}
{
mul.f16x2 r763, r760, r762;
}
{
mul.f16x2 r766, r734, r758;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r734;
mov.b32 r769, {high, low};
}
{
fma.rn.f16x2 r771, r763, r769, r766;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r775, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r777, {high, high};
}
{
mul.f16x2 r779, r640, r777;
}
{
fma.rn.f16x2 r782, r637, r775, r779;
}
{
mul.f16x2 r786, r637, r777;
}
{
neg.f16x2 r789, r786;
}
{
fma.rn.f16x2 r791, r640, r775, r789;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r795, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r797, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r799, {low, high};
}
{
mul.f16x2 r800, r797, r799;
}
{
mul.f16x2 r803, r771, r795;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r771;
mov.b32 r806, {high, low};
}
{
fma.rn.f16x2 r808, r800, r806, r803;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r812, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r814, {high, high};
}
{
mul.f16x2 r816, r652, r814;
}
{
fma.rn.f16x2 r819, r649, r812, r816;
}
{
mul.f16x2 r823, r649, r814;
}
{
neg.f16x2 r826, r823;
}
{
fma.rn.f16x2 r828, r652, r812, r826;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r832, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r834, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r836, {low, high};
}
{
mul.f16x2 r837, r834, r836;
}
{
mul.f16x2 r840, r808, r832;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r808;
mov.b32 r843, {high, low};
}
{
fma.rn.f16x2 r845, r837, r843, r840;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r849, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r851, {high, high};
}
{
mul.f16x2 r853, r598, r851;
}
{
fma.rn.f16x2 r856, r595, r849, r853;
}
{
mul.f16x2 r860, r595, r851;
}
{
neg.f16x2 r863, r860;
}
{
fma.rn.f16x2 r865, r598, r849, r863;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r869, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r871, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r873, {low, high};
}
{
mul.f16x2 r874, r871, r873;
}
{
mul.f16x2 r877, r845, r869;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r845;
mov.b32 r880, {high, low};
}
{
fma.rn.f16x2 r882, r874, r880, r877;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r886, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r888, {high, high};
}
{
mul.f16x2 r890, r610, r888;
}
{
fma.rn.f16x2 r893, r607, r886, r890;
}
{
mul.f16x2 r897, r607, r888;
}
{
neg.f16x2 r900, r897;
}
{
fma.rn.f16x2 r902, r610, r886, r900;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r906, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r908, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r910, {low, high};
}
{
mul.f16x2 r911, r908, r910;
}
{
mul.f16x2 r914, r882, r906;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r882;
mov.b32 r917, {high, low};
}
{
fma.rn.f16x2 r919, r911, r917, r914;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r923, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r925, {high, high};
}
{
mul.f16x2 r927, r622, r925;
}
{
fma.rn.f16x2 r930, r619, r923, r927;
}
{
mul.f16x2 r934, r619, r925;
}
{
neg.f16x2 r937, r934;
}
{
fma.rn.f16x2 r939, r622, r923, r937;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r943, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r945, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r947, {low, high};
}
{
mul.f16x2 r948, r945, r947;
}
{
mul.f16x2 r951, r919, r943;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r919;
mov.b32 r954, {high, low};
}
{
fma.rn.f16x2 r956, r948, r954, r951;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r960, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r962, {high, high};
}
{
mul.f16x2 r964, r634, r962;
}
{
fma.rn.f16x2 r967, r631, r960, r964;
}
{
mul.f16x2 r971, r631, r962;
}
{
neg.f16x2 r974, r971;
}
{
fma.rn.f16x2 r976, r634, r960, r974;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r980, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r982, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r984, {low, high};
}
{
mul.f16x2 r985, r982, r984;
}
{
mul.f16x2 r988, r956, r980;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r956;
mov.b32 r991, {high, low};
}
{
fma.rn.f16x2 r993, r985, r991, r988;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r997, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r999, {high, high};
}
{
mul.f16x2 r1001, r646, r999;
}
{
fma.rn.f16x2 r1004, r643, r997, r1001;
}
{
mul.f16x2 r1008, r643, r999;
}
{
neg.f16x2 r1011, r1008;
}
{
fma.rn.f16x2 r1013, r646, r997, r1011;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r1017, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r661;
mov.b32 r1019, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f123;
cvt.rn.f16.f32 high, f124;
mov.b32 r1021, {low, high};
}
{
mul.f16x2 r1022, r1019, r1021;
}
{
mul.f16x2 r1025, r993, r1017;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r993;
mov.b32 r1028, {high, low};
}
{
fma.rn.f16x2 r1030, r1022, r1028, r1025;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1030;
mov.b32 r1034, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1030;
mov.b32 r1036, {high, high};
}
{
mul.f16x2 r1038, r658, r1036;
}
{
fma.rn.f16x2 r1041, r655, r1034, r1038;
}
{
mul.f16x2 r1045, r655, r1036;
}
{
neg.f16x2 r1048, r1045;
}
{
fma.rn.f16x2 r1050, r658, r1034, r1048;
}
barrier.sync 0;
mad.lo.s32 r1739, r1737, 48, r1738;
st.shared.v4.f32 [r1739], {r589, r671, r708, r745};
st.shared.v4.f32 [r1739+16], {r782, r819, r856, r893};
st.shared.v4.f32 [r1739+32], {r930, r967, r1004, r1041};
barrier.sync 0;
mad.lo.s32 r1740, r1737, -44, r1739;
ld.shared.u32 r1077, [r1740];
ld.shared.u32 r1327, [r1740+48];
ld.shared.u32 r1163, [r1740+96];
ld.shared.u32 r1413, [r1740+144];
ld.shared.u32 r1074, [r1740+192];
ld.shared.u32 r1324, [r1740+240];
ld.shared.u32 r1160, [r1740+288];
ld.shared.u32 r1410, [r1740+336];
ld.shared.u32 r1075, [r1740+384];
ld.shared.u32 r1325, [r1740+432];
ld.shared.u32 r1161, [r1740+480];
ld.shared.u32 r1411, [r1740+528];
barrier.sync 0;
st.shared.v4.f32 [r1739], {r592, r680, r717, r754};
st.shared.v4.f32 [r1739+16], {r791, r828, r865, r902};
st.shared.v4.f32 [r1739+32], {r939, r976, r1013, r1050};
barrier.sync 0;
ld.shared.u32 r1083, [r1740];
ld.shared.u32 r1333, [r1740+48];
ld.shared.u32 r1169, [r1740+96];
ld.shared.u32 r1419, [r1740+144];
ld.shared.u32 r1080, [r1740+192];
ld.shared.u32 r1330, [r1740+240];
ld.shared.u32 r1166, [r1740+288];
ld.shared.u32 r1416, [r1740+336];
ld.shared.u32 r1081, [r1740+384];
ld.shared.u32 r1331, [r1740+432];
ld.shared.u32 r1167, [r1740+480];
ld.shared.u32 r1417, [r1740+528];
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1071, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1072, {low, high};
}
{
add.f16x2 r1073, r1074, r1075;
}
{
add.f16x2 r1076, r1077, r1073;
}
{
add.f16x2 r1079, r1080, r1081;
}
{
add.f16x2 r1082, r1083, r1079;
}
{
add.f16x2 r1085, r1074, r1075;
}
{
mul.f16x2 r1088, r1085, r1071;
}
{
add.f16x2 r1091, r1077, r1088;
}
{
sub.f16x2 r1094, r1080, r1081;
}
{
mul.f16x2 r1097, r1094, r1072;
}
{
add.f16x2 r1100, r1091, r1097;
}
{
add.f16x2 r1103, r1074, r1075;
}
{
mul.f16x2 r1106, r1103, r1071;
}
{
add.f16x2 r1109, r1077, r1106;
}
{
sub.f16x2 r1112, r1080, r1081;
}
{
mul.f16x2 r1115, r1112, r1072;
}
{
sub.f16x2 r1118, r1109, r1115;
}
{
add.f16x2 r1121, r1080, r1081;
}
{
mul.f16x2 r1124, r1121, r1071;
}
{
add.f16x2 r1127, r1083, r1124;
}
{
sub.f16x2 r1130, r1074, r1075;
}
{
mul.f16x2 r1133, r1130, r1072;
}
{
sub.f16x2 r1136, r1127, r1133;
}
{
add.f16x2 r1139, r1080, r1081;
}
{
mul.f16x2 r1142, r1139, r1071;
}
{
add.f16x2 r1145, r1083, r1142;
}
{
sub.f16x2 r1148, r1074, r1075;
}
{
mul.f16x2 r1151, r1148, r1072;
}
{
add.f16x2 r1154, r1145, r1151;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1157, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1158, {low, high};
}
{
add.f16x2 r1159, r1160, r1161;
}
{
add.f16x2 r1162, r1163, r1159;
}
{
add.f16x2 r1165, r1166, r1167;
}
{
add.f16x2 r1168, r1169, r1165;
}
{
add.f16x2 r1171, r1160, r1161;
}
{
mul.f16x2 r1174, r1171, r1157;
}
{
add.f16x2 r1177, r1163, r1174;
}
{
sub.f16x2 r1180, r1166, r1167;
}
{
mul.f16x2 r1183, r1180, r1158;
}
{
add.f16x2 r1186, r1177, r1183;
}
{
add.f16x2 r1189, r1160, r1161;
}
{
mul.f16x2 r1192, r1189, r1157;
}
{
add.f16x2 r1195, r1163, r1192;
}
{
sub.f16x2 r1198, r1166, r1167;
}
{
mul.f16x2 r1201, r1198, r1158;
}
{
sub.f16x2 r1204, r1195, r1201;
}
{
add.f16x2 r1207, r1166, r1167;
}
{
mul.f16x2 r1210, r1207, r1157;
}
{
add.f16x2 r1213, r1169, r1210;
}
{
sub.f16x2 r1216, r1160, r1161;
}
{
mul.f16x2 r1219, r1216, r1158;
}
{
sub.f16x2 r1222, r1213, r1219;
}
{
add.f16x2 r1225, r1166, r1167;
}
{
mul.f16x2 r1228, r1225, r1157;
}
{
add.f16x2 r1231, r1169, r1228;
}
{
sub.f16x2 r1234, r1160, r1161;
}
{
mul.f16x2 r1237, r1234, r1158;
}
{
add.f16x2 r1240, r1231, r1237;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1243, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1244, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1245, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1246, {low, high};
}
{
mul.f16x2 r1253, r1186, r1243;
}
{
mul.f16x2 r1256, r1222, r1244;
}
{
sub.f16x2 r1259, r1253, r1256;
}
{
mul.f16x2 r1262, r1186, r1244;
}
{
fma.rn.f16x2 r1265, r1222, r1243, r1262;
}
{
mul.f16x2 r1269, r1204, r1245;
}
{
mul.f16x2 r1272, r1240, r1246;
}
{
sub.f16x2 r1275, r1269, r1272;
}
{
mul.f16x2 r1278, r1204, r1246;
}
{
fma.rn.f16x2 r1281, r1240, r1245, r1278;
}
{
add.f16x2 r1285, r1076, r1162;
}
{
add.f16x2 r1288, r1082, r1168;
}
{
sub.f16x2 r1291, r1076, r1162;
}
{
sub.f16x2 r1294, r1082, r1168;
}
{
add.f16x2 r1297, r1100, r1259;
}
{
add.f16x2 r1300, r1136, r1265;
}
{
sub.f16x2 r1303, r1100, r1259;
}
{
sub.f16x2 r1306, r1136, r1265;
}
{
add.f16x2 r1309, r1118, r1275;
}
{
add.f16x2 r1312, r1154, r1281;
}
{
sub.f16x2 r1315, r1118, r1275;
}
{
sub.f16x2 r1318, r1154, r1281;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1321, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1322, {low, high};
}
{
add.f16x2 r1323, r1324, r1325;
}
{
add.f16x2 r1326, r1327, r1323;
}
{
add.f16x2 r1329, r1330, r1331;
}
{
add.f16x2 r1332, r1333, r1329;
}
{
add.f16x2 r1335, r1324, r1325;
}
{
mul.f16x2 r1338, r1335, r1321;
}
{
add.f16x2 r1341, r1327, r1338;
}
{
sub.f16x2 r1344, r1330, r1331;
}
{
mul.f16x2 r1347, r1344, r1322;
}
{
add.f16x2 r1350, r1341, r1347;
}
{
add.f16x2 r1353, r1324, r1325;
}
{
mul.f16x2 r1356, r1353, r1321;
}
{
add.f16x2 r1359, r1327, r1356;
}
{
sub.f16x2 r1362, r1330, r1331;
}
{
mul.f16x2 r1365, r1362, r1322;
}
{
sub.f16x2 r1368, r1359, r1365;
}
{
add.f16x2 r1371, r1330, r1331;
}
{
mul.f16x2 r1374, r1371, r1321;
}
{
add.f16x2 r1377, r1333, r1374;
}
{
sub.f16x2 r1380, r1324, r1325;
}
{
mul.f16x2 r1383, r1380, r1322;
}
{
sub.f16x2 r1386, r1377, r1383;
}
{
add.f16x2 r1389, r1330, r1331;
}
{
mul.f16x2 r1392, r1389, r1321;
}
{
add.f16x2 r1395, r1333, r1392;
}
{
sub.f16x2 r1398, r1324, r1325;
}
{
mul.f16x2 r1401, r1398, r1322;
}
{
add.f16x2 r1404, r1395, r1401;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1407, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1408, {low, high};
}
{
add.f16x2 r1409, r1410, r1411;
}
{
add.f16x2 r1412, r1413, r1409;
}
{
add.f16x2 r1415, r1416, r1417;
}
{
add.f16x2 r1418, r1419, r1415;
}
{
add.f16x2 r1421, r1410, r1411;
}
{
mul.f16x2 r1424, r1421, r1407;
}
{
add.f16x2 r1427, r1413, r1424;
}
{
sub.f16x2 r1430, r1416, r1417;
}
{
mul.f16x2 r1433, r1430, r1408;
}
{
add.f16x2 r1436, r1427, r1433;
}
{
add.f16x2 r1439, r1410, r1411;
}
{
mul.f16x2 r1442, r1439, r1407;
}
{
add.f16x2 r1445, r1413, r1442;
}
{
sub.f16x2 r1448, r1416, r1417;
}
{
mul.f16x2 r1451, r1448, r1408;
}
{
sub.f16x2 r1454, r1445, r1451;
}
{
add.f16x2 r1457, r1416, r1417;
}
{
mul.f16x2 r1460, r1457, r1407;
}
{
add.f16x2 r1463, r1419, r1460;
}
{
sub.f16x2 r1466, r1410, r1411;
}
{
mul.f16x2 r1469, r1466, r1408;
}
{
sub.f16x2 r1472, r1463, r1469;
}
{
add.f16x2 r1475, r1416, r1417;
}
{
mul.f16x2 r1478, r1475, r1407;
}
{
add.f16x2 r1481, r1419, r1478;
}
{
sub.f16x2 r1484, r1410, r1411;
}
{
mul.f16x2 r1487, r1484, r1408;
}
{
add.f16x2 r1490, r1481, r1487;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1493, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1494, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1495, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1496, {low, high};
}
{
mul.f16x2 r1503, r1436, r1493;
}
{
mul.f16x2 r1506, r1472, r1494;
}
{
sub.f16x2 r1509, r1503, r1506;
}
{
mul.f16x2 r1512, r1436, r1494;
}
{
fma.rn.f16x2 r1515, r1472, r1493, r1512;
}
{
mul.f16x2 r1519, r1454, r1495;
}
{
mul.f16x2 r1522, r1490, r1496;
}
{
sub.f16x2 r1525, r1519, r1522;
}
{
mul.f16x2 r1528, r1454, r1496;
}
{
fma.rn.f16x2 r1531, r1490, r1495, r1528;
}
{
add.f16x2 r1535, r1326, r1412;
}
{
add.f16x2 r1538, r1332, r1418;
}
{
sub.f16x2 r1541, r1326, r1412;
}
{
sub.f16x2 r1544, r1332, r1418;
}
{
add.f16x2 r1547, r1350, r1509;
}
{
add.f16x2 r1550, r1386, r1515;
}
{
sub.f16x2 r1553, r1350, r1509;
}
{
sub.f16x2 r1556, r1386, r1515;
}
{
add.f16x2 r1559, r1368, r1525;
}
{
add.f16x2 r1562, r1404, r1531;
}
{
sub.f16x2 r1565, r1368, r1525;
}
{
sub.f16x2 r1568, r1404, r1531;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1571, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1572, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1573, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1574, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f196;
cvt.rn.f16.f32 high, f196;
mov.b32 r1577, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f198;
cvt.rn.f16.f32 high, f198;
mov.b32 r1578, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f200;
cvt.rn.f16.f32 high, f200;
mov.b32 r1579, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f202;
cvt.rn.f16.f32 high, f202;
mov.b32 r1580, {low, high};
}
{
mul.f16x2 r1593, r1547, r1571;
}
{
mul.f16x2 r1596, r1550, r1572;
}
{
sub.f16x2 r1599, r1593, r1596;
}
{
mul.f16x2 r1602, r1547, r1572;
}
{
fma.rn.f16x2 r1605, r1550, r1571, r1602;
}
{
mul.f16x2 r1609, r1559, r1573;
}
{
mul.f16x2 r1612, r1562, r1574;
}
{
sub.f16x2 r1615, r1609, r1612;
}
{
mul.f16x2 r1618, r1559, r1574;
}
{
fma.rn.f16x2 r1621, r1562, r1573, r1618;
}
{
neg.f16x2 r1625, r1544;
}
{
mul.f16x2 r1627, r1553, r1577;
}
{
mul.f16x2 r1630, r1556, r1578;
}
{
sub.f16x2 r1633, r1627, r1630;
}
{
mul.f16x2 r1636, r1553, r1578;
}
{
fma.rn.f16x2 r1639, r1556, r1577, r1636;
}
{
mul.f16x2 r1643, r1565, r1579;
}
{
mul.f16x2 r1646, r1568, r1580;
}
{
sub.f16x2 r1649, r1643, r1646;
}
{
mul.f16x2 r1652, r1565, r1580;
}
{
fma.rn.f16x2 r1655, r1568, r1579, r1652;
}
{
add.f16x2 %0, r1285, r1535;
}
{
add.f16x2 %1, r1288, r1538;
}
{
sub.f16x2 %12, r1285, r1535;
}
{
sub.f16x2 %13, r1288, r1538;
}
{
add.f16x2 %2, r1297, r1599;
}
{
add.f16x2 %3, r1300, r1605;
}
{
sub.f16x2 %14, r1297, r1599;
}
{
sub.f16x2 %15, r1300, r1605;
}
{
add.f16x2 %4, r1309, r1615;
}
{
add.f16x2 %5, r1312, r1621;
}
{
sub.f16x2 %16, r1309, r1615;
}
{
sub.f16x2 %17, r1312, r1621;
}
{
add.f16x2 %6, r1291, r1625;
}
{
add.f16x2 %7, r1294, r1541;
}
{
sub.f16x2 %18, r1291, r1625;
}
{
sub.f16x2 %19, r1294, r1541;
}
{
add.f16x2 %8, r1303, r1633;
}
{
add.f16x2 %9, r1306, r1639;
}
{
sub.f16x2 %20, r1303, r1633;
}
{
sub.f16x2 %21, r1306, r1639;
}
{
add.f16x2 %10, r1315, r1649;
}
{
add.f16x2 %11, r1318, r1655;
}
{
sub.f16x2 %22, r1315, r1649;
}
{
sub.f16x2 %23, r1318, r1655;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)));
};


#endif
