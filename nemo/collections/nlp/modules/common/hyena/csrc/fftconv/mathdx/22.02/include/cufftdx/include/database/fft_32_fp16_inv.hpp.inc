#ifndef CUFFTDX_FFT_32_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_32_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<980, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b16 rs<179>;
.reg .b32 r<1583>;
.reg .f64 fd<179>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %64, %72;
}
{
add.f16x2 r4, %65, %73;
}
{
sub.f16x2 r7, %64, %72;
}
{
sub.f16x2 r10, %65, %73;
}
{
add.f16x2 r13, %68, %76;
}
{
add.f16x2 r16, %69, %77;
}
{
sub.f16x2 r19, %68, %76;
}
{
sub.f16x2 r22, %69, %77;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %66, %74;
}
{
add.f16x2 r54, %67, %75;
}
{
sub.f16x2 r57, %66, %74;
}
{
sub.f16x2 r60, %67, %75;
}
{
add.f16x2 r63, %70, %78;
}
{
add.f16x2 r66, %71, %79;
}
{
sub.f16x2 r69, %70, %78;
}
{
sub.f16x2 r72, %71, %79;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f64 fd140, 0d3FE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs1, fd140;
}
{
cvt.rn.f16.f64 rs2, fd140;
}
mov.f64 fd139, 0dBFE6A09E667F3BCD;
{
cvt.rn.f16.f64 rs5, fd139;
}
{
cvt.rn.f16.f64 rs6, fd140;
}
mov.b32 r115, {rs1, rs1};
{
mul.f16x2 r101, r89, r115;
}
mov.b32 r112, {rs2, rs2};
{
mul.f16x2 r104, r92, r112;
}
{
sub.f16x2 r107, r101, r104;
}
{
mul.f16x2 r110, r89, r112;
}
{
fma.rn.f16x2 r113, r92, r115, r110;
}
{
neg.f16x2 r117, r86;
}
mov.b32 r133, {rs5, rs5};
{
mul.f16x2 r119, r95, r133;
}
mov.b32 r130, {rs6, rs6};
{
mul.f16x2 r122, r98, r130;
}
{
sub.f16x2 r125, r119, r122;
}
{
mul.f16x2 r128, r95, r130;
}
{
fma.rn.f16x2 r131, r98, r133, r128;
}
{
add.f16x2 r135, r27, r77;
}
{
add.f16x2 r138, r30, r80;
}
{
sub.f16x2 r141, r27, r77;
}
{
sub.f16x2 r144, r30, r80;
}
{
add.f16x2 r147, r39, r107;
}
{
add.f16x2 r150, r42, r113;
}
{
sub.f16x2 r153, r39, r107;
}
{
sub.f16x2 r156, r42, r113;
}
{
add.f16x2 r159, r33, r117;
}
{
add.f16x2 r162, r36, r83;
}
{
sub.f16x2 r165, r33, r117;
}
{
sub.f16x2 r168, r36, r83;
}
{
add.f16x2 r171, r45, r125;
}
{
add.f16x2 r174, r48, r131;
}
{
sub.f16x2 r177, r45, r125;
}
{
sub.f16x2 r180, r48, r131;
}
{
add.f16x2 r183, %80, %92;
}
{
add.f16x2 r186, %82, %94;
}
{
sub.f16x2 r189, %80, %92;
}
{
sub.f16x2 r192, %82, %94;
}
{
add.f16x2 r195, %88, %84;
}
{
add.f16x2 r198, %90, %86;
}
{
sub.f16x2 r201, %88, %84;
}
{
sub.f16x2 r204, %90, %86;
}
{
neg.f16x2 r207, r204;
}
{
add.f16x2 r209, r183, r195;
}
{
add.f16x2 r212, r186, r198;
}
{
sub.f16x2 r215, r183, r195;
}
{
sub.f16x2 r218, r186, r198;
}
{
add.f16x2 r221, r189, r207;
}
{
add.f16x2 r224, r192, r201;
}
{
sub.f16x2 r227, r189, r207;
}
{
sub.f16x2 r230, r192, r201;
}
{
add.f16x2 r233, %89, %85;
}
{
add.f16x2 r236, %91, %87;
}
{
sub.f16x2 r239, %89, %85;
}
{
sub.f16x2 r242, %91, %87;
}
{
add.f16x2 r245, %81, %93;
}
{
add.f16x2 r248, %83, %95;
}
{
sub.f16x2 r251, %81, %93;
}
{
sub.f16x2 r254, %83, %95;
}
{
neg.f16x2 r257, r254;
}
{
add.f16x2 r259, r233, r245;
}
{
add.f16x2 r262, r236, r248;
}
{
sub.f16x2 r265, r233, r245;
}
{
sub.f16x2 r268, r236, r248;
}
{
add.f16x2 r271, r239, r257;
}
{
add.f16x2 r274, r242, r251;
}
{
sub.f16x2 r277, r239, r257;
}
{
sub.f16x2 r280, r242, r251;
}
{
cvt.rn.f16.f64 rs15, fd140;
}
{
cvt.rn.f16.f64 rs16, fd140;
}
{
cvt.rn.f16.f64 rs19, fd139;
}
{
cvt.rn.f16.f64 rs20, fd140;
}
mov.b32 r297, {rs15, rs15};
{
mul.f16x2 r283, r271, r297;
}
mov.b32 r294, {rs16, rs16};
{
mul.f16x2 r286, r274, r294;
}
{
sub.f16x2 r289, r283, r286;
}
{
mul.f16x2 r292, r271, r294;
}
{
fma.rn.f16x2 r295, r274, r297, r292;
}
{
neg.f16x2 r299, r268;
}
mov.b32 r315, {rs19, rs19};
{
mul.f16x2 r301, r277, r315;
}
mov.b32 r312, {rs20, rs20};
{
mul.f16x2 r304, r280, r312;
}
{
sub.f16x2 r307, r301, r304;
}
{
mul.f16x2 r310, r277, r312;
}
{
fma.rn.f16x2 r313, r280, r315, r310;
}
{
add.f16x2 r317, r209, r259;
}
{
add.f16x2 r320, r212, r262;
}
{
sub.f16x2 r323, r209, r259;
}
{
sub.f16x2 r326, r212, r262;
}
{
add.f16x2 r329, r221, r289;
}
{
add.f16x2 r332, r224, r295;
}
{
sub.f16x2 r335, r221, r289;
}
{
sub.f16x2 r338, r224, r295;
}
{
add.f16x2 r341, r215, r299;
}
{
add.f16x2 r344, r218, r265;
}
{
sub.f16x2 r347, r215, r299;
}
{
sub.f16x2 r350, r218, r265;
}
{
add.f16x2 r353, r227, r307;
}
{
add.f16x2 r356, r230, r313;
}
{
sub.f16x2 r359, r227, r307;
}
{
sub.f16x2 r362, r230, r313;
}
mov.f64 fd136, 0d3FED906BCF328D46;
{
cvt.rn.f16.f64 rs29, fd136;
}
mov.f64 fd144, 0d3FD87DE2A6AEA963;
{
cvt.rn.f16.f64 rs30, fd144;
}
{
cvt.rn.f16.f64 rs31, fd140;
}
{
cvt.rn.f16.f64 rs32, fd140;
}
{
cvt.rn.f16.f64 rs33, fd144;
}
{
cvt.rn.f16.f64 rs34, fd136;
}
mov.f64 fd135, 0dBFD87DE2A6AEA963;
{
cvt.rn.f16.f64 rs37, fd135;
}
{
cvt.rn.f16.f64 rs38, fd136;
}
{
cvt.rn.f16.f64 rs39, fd139;
}
{
cvt.rn.f16.f64 rs40, fd140;
}
mov.f64 fd143, 0dBFED906BCF328D46;
{
cvt.rn.f16.f64 rs41, fd143;
}
{
cvt.rn.f16.f64 rs42, fd144;
}
mov.b32 r379, {rs29, rs29};
{
mul.f16x2 r365, r329, r379;
}
mov.b32 r376, {rs30, rs30};
{
mul.f16x2 r368, r332, r376;
}
{
sub.f16x2 r371, r365, r368;
}
{
mul.f16x2 r374, r329, r376;
}
{
fma.rn.f16x2 r377, r332, r379, r374;
}
mov.b32 r395, {rs31, rs31};
{
mul.f16x2 r381, r341, r395;
}
mov.b32 r392, {rs32, rs32};
{
mul.f16x2 r384, r344, r392;
}
{
sub.f16x2 r387, r381, r384;
}
{
mul.f16x2 r390, r341, r392;
}
{
fma.rn.f16x2 r393, r344, r395, r390;
}
mov.b32 r411, {rs33, rs33};
{
mul.f16x2 r397, r353, r411;
}
mov.b32 r408, {rs34, rs34};
{
mul.f16x2 r400, r356, r408;
}
{
sub.f16x2 r403, r397, r400;
}
{
mul.f16x2 r406, r353, r408;
}
{
fma.rn.f16x2 r409, r356, r411, r406;
}
{
neg.f16x2 r413, r326;
}
mov.b32 r429, {rs37, rs37};
{
mul.f16x2 r415, r335, r429;
}
mov.b32 r426, {rs38, rs38};
{
mul.f16x2 r418, r338, r426;
}
{
sub.f16x2 r421, r415, r418;
}
{
mul.f16x2 r424, r335, r426;
}
{
fma.rn.f16x2 r427, r338, r429, r424;
}
mov.b32 r445, {rs39, rs39};
{
mul.f16x2 r431, r347, r445;
}
mov.b32 r442, {rs40, rs40};
{
mul.f16x2 r434, r350, r442;
}
{
sub.f16x2 r437, r431, r434;
}
{
mul.f16x2 r440, r347, r442;
}
{
fma.rn.f16x2 r443, r350, r445, r440;
}
mov.b32 r461, {rs41, rs41};
{
mul.f16x2 r447, r359, r461;
}
mov.b32 r458, {rs42, rs42};
{
mul.f16x2 r450, r362, r458;
}
{
sub.f16x2 r453, r447, r450;
}
{
mul.f16x2 r456, r359, r458;
}
{
fma.rn.f16x2 r459, r362, r461, r456;
}
{
add.f16x2 r463, r135, r317;
}
{
add.f16x2 r466, r138, r320;
}
{
sub.f16x2 r469, r135, r317;
}
{
sub.f16x2 r472, r138, r320;
}
{
add.f16x2 r475, r147, r371;
}
{
add.f16x2 r478, r150, r377;
}
{
sub.f16x2 r481, r147, r371;
}
{
sub.f16x2 r484, r150, r377;
}
{
add.f16x2 r487, r159, r387;
}
{
add.f16x2 r490, r162, r393;
}
{
sub.f16x2 r493, r159, r387;
}
{
sub.f16x2 r496, r162, r393;
}
{
add.f16x2 r499, r171, r403;
}
{
add.f16x2 r502, r174, r409;
}
{
sub.f16x2 r505, r171, r403;
}
{
sub.f16x2 r508, r174, r409;
}
{
add.f16x2 r511, r141, r413;
}
{
add.f16x2 r514, r144, r323;
}
{
sub.f16x2 r517, r141, r413;
}
{
sub.f16x2 r520, r144, r323;
}
{
add.f16x2 r523, r153, r421;
}
{
add.f16x2 r526, r156, r427;
}
{
sub.f16x2 r529, r153, r421;
}
{
sub.f16x2 r532, r156, r427;
}
{
add.f16x2 r535, r165, r437;
}
{
add.f16x2 r538, r168, r443;
}
{
sub.f16x2 r541, r165, r437;
}
{
sub.f16x2 r544, r168, r443;
}
{
add.f16x2 r547, r177, r453;
}
{
add.f16x2 r550, r180, r459;
}
{
sub.f16x2 r553, r177, r453;
}
{
sub.f16x2 r556, r180, r459;
}
{
add.f16x2 r559, %110, %106;
}
{
add.f16x2 r562, %96, %108;
}
{
sub.f16x2 r565, %110, %106;
}
{
sub.f16x2 r568, %96, %108;
}
{
add.f16x2 r571, %102, %98;
}
{
add.f16x2 r574, %104, %100;
}
{
sub.f16x2 r577, %102, %98;
}
{
sub.f16x2 r580, %104, %100;
}
{
neg.f16x2 r583, r580;
}
{
add.f16x2 r585, r559, r571;
}
{
add.f16x2 r588, r562, r574;
}
{
sub.f16x2 r591, r559, r571;
}
{
sub.f16x2 r594, r562, r574;
}
{
add.f16x2 r597, r565, r583;
}
{
add.f16x2 r600, r568, r577;
}
{
sub.f16x2 r603, r565, r583;
}
{
sub.f16x2 r606, r568, r577;
}
{
add.f16x2 r609, %103, %99;
}
{
add.f16x2 r612, %105, %101;
}
{
sub.f16x2 r615, %103, %99;
}
{
sub.f16x2 r618, %105, %101;
}
{
add.f16x2 r621, %111, %107;
}
{
add.f16x2 r624, %97, %109;
}
{
sub.f16x2 r627, %111, %107;
}
{
sub.f16x2 r630, %97, %109;
}
{
neg.f16x2 r633, r630;
}
{
add.f16x2 r635, r609, r621;
}
{
add.f16x2 r638, r612, r624;
}
{
sub.f16x2 r641, r609, r621;
}
{
sub.f16x2 r644, r612, r624;
}
{
add.f16x2 r647, r615, r633;
}
{
add.f16x2 r650, r618, r627;
}
{
sub.f16x2 r653, r615, r633;
}
{
sub.f16x2 r656, r618, r627;
}
{
cvt.rn.f16.f64 rs59, fd140;
}
{
cvt.rn.f16.f64 rs60, fd140;
}
{
cvt.rn.f16.f64 rs63, fd139;
}
{
cvt.rn.f16.f64 rs64, fd140;
}
mov.b32 r673, {rs59, rs59};
{
mul.f16x2 r659, r647, r673;
}
mov.b32 r670, {rs60, rs60};
{
mul.f16x2 r662, r650, r670;
}
{
sub.f16x2 r665, r659, r662;
}
{
mul.f16x2 r668, r647, r670;
}
{
fma.rn.f16x2 r671, r650, r673, r668;
}
{
neg.f16x2 r675, r644;
}
mov.b32 r691, {rs63, rs63};
{
mul.f16x2 r677, r653, r691;
}
mov.b32 r688, {rs64, rs64};
{
mul.f16x2 r680, r656, r688;
}
{
sub.f16x2 r683, r677, r680;
}
{
mul.f16x2 r686, r653, r688;
}
{
fma.rn.f16x2 r689, r656, r691, r686;
}
{
add.f16x2 r693, r585, r635;
}
{
add.f16x2 r696, r588, r638;
}
{
sub.f16x2 r699, r585, r635;
}
{
sub.f16x2 r702, r588, r638;
}
{
add.f16x2 r705, r597, r665;
}
{
add.f16x2 r708, r600, r671;
}
{
sub.f16x2 r711, r597, r665;
}
{
sub.f16x2 r714, r600, r671;
}
{
add.f16x2 r717, r591, r675;
}
{
add.f16x2 r720, r594, r641;
}
{
sub.f16x2 r723, r591, r675;
}
{
sub.f16x2 r726, r594, r641;
}
{
add.f16x2 r729, r603, r683;
}
{
add.f16x2 r732, r606, r689;
}
{
sub.f16x2 r735, r603, r683;
}
{
sub.f16x2 r738, r606, r689;
}
{
add.f16x2 r741, %114, %126;
}
{
add.f16x2 r744, %116, %112;
}
{
sub.f16x2 r747, %114, %126;
}
{
sub.f16x2 r750, %116, %112;
}
{
add.f16x2 r753, %122, %118;
}
{
add.f16x2 r756, %124, %120;
}
{
sub.f16x2 r759, %122, %118;
}
{
sub.f16x2 r762, %124, %120;
}
{
neg.f16x2 r765, r762;
}
{
add.f16x2 r767, r741, r753;
}
{
add.f16x2 r770, r744, r756;
}
{
sub.f16x2 r773, r741, r753;
}
{
sub.f16x2 r776, r744, r756;
}
{
add.f16x2 r779, r747, r765;
}
{
add.f16x2 r782, r750, r759;
}
{
sub.f16x2 r785, r747, r765;
}
{
sub.f16x2 r788, r750, r759;
}
{
add.f16x2 r791, %123, %119;
}
{
add.f16x2 r794, %125, %121;
}
{
sub.f16x2 r797, %123, %119;
}
{
sub.f16x2 r800, %125, %121;
}
{
add.f16x2 r803, %115, %127;
}
{
add.f16x2 r806, %117, %113;
}
{
sub.f16x2 r809, %115, %127;
}
{
sub.f16x2 r812, %117, %113;
}
{
neg.f16x2 r815, r812;
}
{
add.f16x2 r817, r791, r803;
}
{
add.f16x2 r820, r794, r806;
}
{
sub.f16x2 r823, r791, r803;
}
{
sub.f16x2 r826, r794, r806;
}
{
add.f16x2 r829, r797, r815;
}
{
add.f16x2 r832, r800, r809;
}
{
sub.f16x2 r835, r797, r815;
}
{
sub.f16x2 r838, r800, r809;
}
{
cvt.rn.f16.f64 rs73, fd140;
}
{
cvt.rn.f16.f64 rs74, fd140;
}
{
cvt.rn.f16.f64 rs77, fd139;
}
{
cvt.rn.f16.f64 rs78, fd140;
}
mov.b32 r855, {rs73, rs73};
{
mul.f16x2 r841, r829, r855;
}
mov.b32 r852, {rs74, rs74};
{
mul.f16x2 r844, r832, r852;
}
{
sub.f16x2 r847, r841, r844;
}
{
mul.f16x2 r850, r829, r852;
}
{
fma.rn.f16x2 r853, r832, r855, r850;
}
{
neg.f16x2 r857, r826;
}
mov.b32 r873, {rs77, rs77};
{
mul.f16x2 r859, r835, r873;
}
mov.b32 r870, {rs78, rs78};
{
mul.f16x2 r862, r838, r870;
}
{
sub.f16x2 r865, r859, r862;
}
{
mul.f16x2 r868, r835, r870;
}
{
fma.rn.f16x2 r871, r838, r873, r868;
}
{
add.f16x2 r875, r767, r817;
}
{
add.f16x2 r878, r770, r820;
}
{
sub.f16x2 r881, r767, r817;
}
{
sub.f16x2 r884, r770, r820;
}
{
add.f16x2 r887, r779, r847;
}
{
add.f16x2 r890, r782, r853;
}
{
sub.f16x2 r893, r779, r847;
}
{
sub.f16x2 r896, r782, r853;
}
{
add.f16x2 r899, r773, r857;
}
{
add.f16x2 r902, r776, r823;
}
{
sub.f16x2 r905, r773, r857;
}
{
sub.f16x2 r908, r776, r823;
}
{
add.f16x2 r911, r785, r865;
}
{
add.f16x2 r914, r788, r871;
}
{
sub.f16x2 r917, r785, r865;
}
{
sub.f16x2 r920, r788, r871;
}
{
cvt.rn.f16.f64 rs87, fd136;
}
{
cvt.rn.f16.f64 rs88, fd144;
}
{
cvt.rn.f16.f64 rs89, fd140;
}
{
cvt.rn.f16.f64 rs90, fd140;
}
{
cvt.rn.f16.f64 rs91, fd144;
}
{
cvt.rn.f16.f64 rs92, fd136;
}
{
cvt.rn.f16.f64 rs95, fd135;
}
{
cvt.rn.f16.f64 rs96, fd136;
}
{
cvt.rn.f16.f64 rs97, fd139;
}
{
cvt.rn.f16.f64 rs98, fd140;
}
{
cvt.rn.f16.f64 rs99, fd143;
}
{
cvt.rn.f16.f64 rs100, fd144;
}
mov.b32 r937, {rs87, rs87};
{
mul.f16x2 r923, r887, r937;
}
mov.b32 r934, {rs88, rs88};
{
mul.f16x2 r926, r890, r934;
}
{
sub.f16x2 r929, r923, r926;
}
{
mul.f16x2 r932, r887, r934;
}
{
fma.rn.f16x2 r935, r890, r937, r932;
}
mov.b32 r953, {rs89, rs89};
{
mul.f16x2 r939, r899, r953;
}
mov.b32 r950, {rs90, rs90};
{
mul.f16x2 r942, r902, r950;
}
{
sub.f16x2 r945, r939, r942;
}
{
mul.f16x2 r948, r899, r950;
}
{
fma.rn.f16x2 r951, r902, r953, r948;
}
mov.b32 r969, {rs91, rs91};
{
mul.f16x2 r955, r911, r969;
}
mov.b32 r966, {rs92, rs92};
{
mul.f16x2 r958, r914, r966;
}
{
sub.f16x2 r961, r955, r958;
}
{
mul.f16x2 r964, r911, r966;
}
{
fma.rn.f16x2 r967, r914, r969, r964;
}
{
neg.f16x2 r971, r884;
}
mov.b32 r987, {rs95, rs95};
{
mul.f16x2 r973, r893, r987;
}
mov.b32 r984, {rs96, rs96};
{
mul.f16x2 r976, r896, r984;
}
{
sub.f16x2 r979, r973, r976;
}
{
mul.f16x2 r982, r893, r984;
}
{
fma.rn.f16x2 r985, r896, r987, r982;
}
mov.b32 r1003, {rs97, rs97};
{
mul.f16x2 r989, r905, r1003;
}
mov.b32 r1000, {rs98, rs98};
{
mul.f16x2 r992, r908, r1000;
}
{
sub.f16x2 r995, r989, r992;
}
{
mul.f16x2 r998, r905, r1000;
}
{
fma.rn.f16x2 r1001, r908, r1003, r998;
}
mov.b32 r1019, {rs99, rs99};
{
mul.f16x2 r1005, r917, r1019;
}
mov.b32 r1016, {rs100, rs100};
{
mul.f16x2 r1008, r920, r1016;
}
{
sub.f16x2 r1011, r1005, r1008;
}
{
mul.f16x2 r1014, r917, r1016;
}
{
fma.rn.f16x2 r1017, r920, r1019, r1014;
}
{
add.f16x2 r1021, r693, r875;
}
{
add.f16x2 r1024, r696, r878;
}
{
sub.f16x2 r1027, r693, r875;
}
{
sub.f16x2 r1030, r696, r878;
}
{
add.f16x2 r1033, r705, r929;
}
{
add.f16x2 r1036, r708, r935;
}
{
sub.f16x2 r1039, r705, r929;
}
{
sub.f16x2 r1042, r708, r935;
}
{
add.f16x2 r1045, r717, r945;
}
{
add.f16x2 r1048, r720, r951;
}
{
sub.f16x2 r1051, r717, r945;
}
{
sub.f16x2 r1054, r720, r951;
}
{
add.f16x2 r1057, r729, r961;
}
{
add.f16x2 r1060, r732, r967;
}
{
sub.f16x2 r1063, r729, r961;
}
{
sub.f16x2 r1066, r732, r967;
}
{
add.f16x2 r1069, r699, r971;
}
{
add.f16x2 r1072, r702, r881;
}
{
sub.f16x2 r1075, r699, r971;
}
{
sub.f16x2 r1078, r702, r881;
}
{
add.f16x2 r1081, r711, r979;
}
{
add.f16x2 r1084, r714, r985;
}
{
sub.f16x2 r1087, r711, r979;
}
{
sub.f16x2 r1090, r714, r985;
}
{
add.f16x2 r1093, r723, r995;
}
{
add.f16x2 r1096, r726, r1001;
}
{
sub.f16x2 r1099, r723, r995;
}
{
sub.f16x2 r1102, r726, r1001;
}
{
add.f16x2 r1105, r735, r1011;
}
{
add.f16x2 r1108, r738, r1017;
}
{
sub.f16x2 r1111, r735, r1011;
}
{
sub.f16x2 r1114, r738, r1017;
}
mov.f64 fd134, 0d3FEF6297CFF75CB0;
{
cvt.rn.f16.f64 rs117, fd134;
}
mov.f64 fd146, 0d3FC8F8B83C69A60B;
{
cvt.rn.f16.f64 rs118, fd146;
}
{
cvt.rn.f16.f64 rs119, fd136;
}
{
cvt.rn.f16.f64 rs120, fd144;
}
mov.f64 fd138, 0d3FEA9B66290EA1A3;
{
cvt.rn.f16.f64 rs121, fd138;
}
mov.f64 fd142, 0d3FE1C73B39AE68C8;
{
cvt.rn.f16.f64 rs122, fd142;
}
{
cvt.rn.f16.f64 rs123, fd140;
}
{
cvt.rn.f16.f64 rs124, fd140;
}
{
cvt.rn.f16.f64 rs125, fd142;
}
{
cvt.rn.f16.f64 rs126, fd138;
}
{
cvt.rn.f16.f64 rs127, fd144;
}
{
cvt.rn.f16.f64 rs128, fd136;
}
{
cvt.rn.f16.f64 rs129, fd146;
}
{
cvt.rn.f16.f64 rs130, fd134;
}
mov.f64 fd133, 0dBFC8F8B83C69A60B;
{
cvt.rn.f16.f64 rs133, fd133;
}
{
cvt.rn.f16.f64 rs134, fd134;
}
{
cvt.rn.f16.f64 rs135, fd135;
}
{
cvt.rn.f16.f64 rs136, fd136;
}
mov.f64 fd137, 0dBFE1C73B39AE68C8;
{
cvt.rn.f16.f64 rs137, fd137;
}
{
cvt.rn.f16.f64 rs138, fd138;
}
{
cvt.rn.f16.f64 rs139, fd139;
}
{
cvt.rn.f16.f64 rs140, fd140;
}
mov.f64 fd141, 0dBFEA9B66290EA1A3;
{
cvt.rn.f16.f64 rs141, fd141;
}
{
cvt.rn.f16.f64 rs142, fd142;
}
{
cvt.rn.f16.f64 rs143, fd143;
}
{
cvt.rn.f16.f64 rs144, fd144;
}
mov.f64 fd145, 0dBFEF6297CFF75CB0;
{
cvt.rn.f16.f64 rs145, fd145;
}
{
cvt.rn.f16.f64 rs146, fd146;
}
mov.b32 r1131, {rs117, rs117};
{
mul.f16x2 r1117, r1033, r1131;
}
mov.b32 r1128, {rs118, rs118};
{
mul.f16x2 r1120, r1036, r1128;
}
{
sub.f16x2 r1123, r1117, r1120;
}
{
mul.f16x2 r1126, r1033, r1128;
}
{
fma.rn.f16x2 r1129, r1036, r1131, r1126;
}
mov.b32 r1147, {rs119, rs119};
{
mul.f16x2 r1133, r1045, r1147;
}
mov.b32 r1144, {rs120, rs120};
{
mul.f16x2 r1136, r1048, r1144;
}
{
sub.f16x2 r1139, r1133, r1136;
}
{
mul.f16x2 r1142, r1045, r1144;
}
{
fma.rn.f16x2 r1145, r1048, r1147, r1142;
}
mov.b32 r1163, {rs121, rs121};
{
mul.f16x2 r1149, r1057, r1163;
}
mov.b32 r1160, {rs122, rs122};
{
mul.f16x2 r1152, r1060, r1160;
}
{
sub.f16x2 r1155, r1149, r1152;
}
{
mul.f16x2 r1158, r1057, r1160;
}
{
fma.rn.f16x2 r1161, r1060, r1163, r1158;
}
mov.b32 r1179, {rs123, rs123};
{
mul.f16x2 r1165, r1069, r1179;
}
mov.b32 r1176, {rs124, rs124};
{
mul.f16x2 r1168, r1072, r1176;
}
{
sub.f16x2 r1171, r1165, r1168;
}
{
mul.f16x2 r1174, r1069, r1176;
}
{
fma.rn.f16x2 r1177, r1072, r1179, r1174;
}
mov.b32 r1195, {rs125, rs125};
{
mul.f16x2 r1181, r1081, r1195;
}
mov.b32 r1192, {rs126, rs126};
{
mul.f16x2 r1184, r1084, r1192;
}
{
sub.f16x2 r1187, r1181, r1184;
}
{
mul.f16x2 r1190, r1081, r1192;
}
{
fma.rn.f16x2 r1193, r1084, r1195, r1190;
}
mov.b32 r1211, {rs127, rs127};
{
mul.f16x2 r1197, r1093, r1211;
}
mov.b32 r1208, {rs128, rs128};
{
mul.f16x2 r1200, r1096, r1208;
}
{
sub.f16x2 r1203, r1197, r1200;
}
{
mul.f16x2 r1206, r1093, r1208;
}
{
fma.rn.f16x2 r1209, r1096, r1211, r1206;
}
mov.b32 r1227, {rs129, rs129};
{
mul.f16x2 r1213, r1105, r1227;
}
mov.b32 r1224, {rs130, rs130};
{
mul.f16x2 r1216, r1108, r1224;
}
{
sub.f16x2 r1219, r1213, r1216;
}
{
mul.f16x2 r1222, r1105, r1224;
}
{
fma.rn.f16x2 r1225, r1108, r1227, r1222;
}
{
neg.f16x2 r1229, r1030;
}
mov.b32 r1245, {rs133, rs133};
{
mul.f16x2 r1231, r1039, r1245;
}
mov.b32 r1242, {rs134, rs134};
{
mul.f16x2 r1234, r1042, r1242;
}
{
sub.f16x2 r1237, r1231, r1234;
}
{
mul.f16x2 r1240, r1039, r1242;
}
{
fma.rn.f16x2 r1243, r1042, r1245, r1240;
}
mov.b32 r1261, {rs135, rs135};
{
mul.f16x2 r1247, r1051, r1261;
}
mov.b32 r1258, {rs136, rs136};
{
mul.f16x2 r1250, r1054, r1258;
}
{
sub.f16x2 r1253, r1247, r1250;
}
{
mul.f16x2 r1256, r1051, r1258;
}
{
fma.rn.f16x2 r1259, r1054, r1261, r1256;
}
mov.b32 r1277, {rs137, rs137};
{
mul.f16x2 r1263, r1063, r1277;
}
mov.b32 r1274, {rs138, rs138};
{
mul.f16x2 r1266, r1066, r1274;
}
{
sub.f16x2 r1269, r1263, r1266;
}
{
mul.f16x2 r1272, r1063, r1274;
}
{
fma.rn.f16x2 r1275, r1066, r1277, r1272;
}
mov.b32 r1293, {rs139, rs139};
{
mul.f16x2 r1279, r1075, r1293;
}
mov.b32 r1290, {rs140, rs140};
{
mul.f16x2 r1282, r1078, r1290;
}
{
sub.f16x2 r1285, r1279, r1282;
}
{
mul.f16x2 r1288, r1075, r1290;
}
{
fma.rn.f16x2 r1291, r1078, r1293, r1288;
}
mov.b32 r1309, {rs141, rs141};
{
mul.f16x2 r1295, r1087, r1309;
}
mov.b32 r1306, {rs142, rs142};
{
mul.f16x2 r1298, r1090, r1306;
}
{
sub.f16x2 r1301, r1295, r1298;
}
{
mul.f16x2 r1304, r1087, r1306;
}
{
fma.rn.f16x2 r1307, r1090, r1309, r1304;
}
mov.b32 r1325, {rs143, rs143};
{
mul.f16x2 r1311, r1099, r1325;
}
mov.b32 r1322, {rs144, rs144};
{
mul.f16x2 r1314, r1102, r1322;
}
{
sub.f16x2 r1317, r1311, r1314;
}
{
mul.f16x2 r1320, r1099, r1322;
}
{
fma.rn.f16x2 r1323, r1102, r1325, r1320;
}
mov.b32 r1341, {rs145, rs145};
{
mul.f16x2 r1327, r1111, r1341;
}
mov.b32 r1338, {rs146, rs146};
{
mul.f16x2 r1330, r1114, r1338;
}
{
sub.f16x2 r1333, r1327, r1330;
}
{
mul.f16x2 r1336, r1111, r1338;
}
{
fma.rn.f16x2 r1339, r1114, r1341, r1336;
}
{
add.f16x2 %0, r463, r1021;
}
{
add.f16x2 %1, r466, r1024;
}
{
sub.f16x2 %32, r463, r1021;
}
{
sub.f16x2 %33, r466, r1024;
}
{
add.f16x2 %2, r475, r1123;
}
{
add.f16x2 %3, r478, r1129;
}
{
sub.f16x2 %34, r475, r1123;
}
{
sub.f16x2 %35, r478, r1129;
}
{
add.f16x2 %4, r487, r1139;
}
{
add.f16x2 %5, r490, r1145;
}
{
sub.f16x2 %36, r487, r1139;
}
{
sub.f16x2 %37, r490, r1145;
}
{
add.f16x2 %6, r499, r1155;
}
{
add.f16x2 %7, r502, r1161;
}
{
sub.f16x2 %38, r499, r1155;
}
{
sub.f16x2 %39, r502, r1161;
}
{
add.f16x2 %8, r511, r1171;
}
{
add.f16x2 %9, r514, r1177;
}
{
sub.f16x2 %40, r511, r1171;
}
{
sub.f16x2 %41, r514, r1177;
}
{
add.f16x2 %10, r523, r1187;
}
{
add.f16x2 %11, r526, r1193;
}
{
sub.f16x2 %42, r523, r1187;
}
{
sub.f16x2 %43, r526, r1193;
}
{
add.f16x2 %12, r535, r1203;
}
{
add.f16x2 %13, r538, r1209;
}
{
sub.f16x2 %44, r535, r1203;
}
{
sub.f16x2 %45, r538, r1209;
}
{
add.f16x2 %14, r547, r1219;
}
{
add.f16x2 %15, r550, r1225;
}
{
sub.f16x2 %46, r547, r1219;
}
{
sub.f16x2 %47, r550, r1225;
}
{
add.f16x2 %16, r469, r1229;
}
{
add.f16x2 %17, r472, r1027;
}
{
sub.f16x2 %48, r469, r1229;
}
{
sub.f16x2 %49, r472, r1027;
}
{
add.f16x2 %18, r481, r1237;
}
{
add.f16x2 %19, r484, r1243;
}
{
sub.f16x2 %50, r481, r1237;
}
{
sub.f16x2 %51, r484, r1243;
}
{
add.f16x2 %20, r493, r1253;
}
{
add.f16x2 %21, r496, r1259;
}
{
sub.f16x2 %52, r493, r1253;
}
{
sub.f16x2 %53, r496, r1259;
}
{
add.f16x2 %22, r505, r1269;
}
{
add.f16x2 %23, r508, r1275;
}
{
sub.f16x2 %54, r505, r1269;
}
{
sub.f16x2 %55, r508, r1275;
}
{
add.f16x2 %24, r517, r1285;
}
{
add.f16x2 %25, r520, r1291;
}
{
sub.f16x2 %56, r517, r1285;
}
{
sub.f16x2 %57, r520, r1291;
}
{
add.f16x2 %26, r529, r1301;
}
{
add.f16x2 %27, r532, r1307;
}
{
sub.f16x2 %58, r529, r1301;
}
{
sub.f16x2 %59, r532, r1307;
}
{
add.f16x2 %28, r541, r1317;
}
{
add.f16x2 %29, r544, r1323;
}
{
sub.f16x2 %60, r541, r1317;
}
{
sub.f16x2 %61, r544, r1323;
}
{
add.f16x2 %30, r553, r1333;
}
{
add.f16x2 %31, r556, r1339;
}
{
sub.f16x2 %62, r553, r1333;
}
{
sub.f16x2 %63, r556, r1339;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)), "=r"(__HALF2_TO_UI(rmem[16].x)), "=r"(__HALF2_TO_UI(rmem[16].y)), "=r"(__HALF2_TO_UI(rmem[17].x)), "=r"(__HALF2_TO_UI(rmem[17].y)), "=r"(__HALF2_TO_UI(rmem[18].x)), "=r"(__HALF2_TO_UI(rmem[18].y)), "=r"(__HALF2_TO_UI(rmem[19].x)), "=r"(__HALF2_TO_UI(rmem[19].y)), "=r"(__HALF2_TO_UI(rmem[20].x)), "=r"(__HALF2_TO_UI(rmem[20].y)), "=r"(__HALF2_TO_UI(rmem[21].x)), "=r"(__HALF2_TO_UI(rmem[21].y)), "=r"(__HALF2_TO_UI(rmem[22].x)), "=r"(__HALF2_TO_UI(rmem[22].y)), "=r"(__HALF2_TO_UI(rmem[23].x)), "=r"(__HALF2_TO_UI(rmem[23].y)), "=r"(__HALF2_TO_UI(rmem[24].x)), "=r"(__HALF2_TO_UI(rmem[24].y)), "=r"(__HALF2_TO_UI(rmem[25].x)), "=r"(__HALF2_TO_UI(rmem[25].y)), "=r"(__HALF2_TO_UI(rmem[26].x)), "=r"(__HALF2_TO_UI(rmem[26].y)), "=r"(__HALF2_TO_UI(rmem[27].x)), "=r"(__HALF2_TO_UI(rmem[27].y)), "=r"(__HALF2_TO_UI(rmem[28].x)), "=r"(__HALF2_TO_UI(rmem[28].y)), "=r"(__HALF2_TO_UI(rmem[29].x)), "=r"(__HALF2_TO_UI(rmem[29].y)), "=r"(__HALF2_TO_UI(rmem[30].x)), "=r"(__HALF2_TO_UI(rmem[30].y)), "=r"(__HALF2_TO_UI(rmem[31].x)), "=r"(__HALF2_TO_UI(rmem[31].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[16].x)), "r"(__HALF2_TO_UI(rmem[16].y)), "r"(__HALF2_TO_UI(rmem[20].x)), "r"(__HALF2_TO_UI(rmem[20].y)), "r"(__HALF2_TO_UI(rmem[24].x)), "r"(__HALF2_TO_UI(rmem[24].y)), "r"(__HALF2_TO_UI(rmem[28].x)), "r"(__HALF2_TO_UI(rmem[28].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[26].x)), "r"(__HALF2_TO_UI(rmem[22].x)), "r"(__HALF2_TO_UI(rmem[26].y)), "r"(__HALF2_TO_UI(rmem[22].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[18].x)), "r"(__HALF2_TO_UI(rmem[30].x)), "r"(__HALF2_TO_UI(rmem[18].y)), "r"(__HALF2_TO_UI(rmem[30].y)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[25].x)), "r"(__HALF2_TO_UI(rmem[21].x)), "r"(__HALF2_TO_UI(rmem[25].y)), "r"(__HALF2_TO_UI(rmem[21].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[17].x)), "r"(__HALF2_TO_UI(rmem[29].x)), "r"(__HALF2_TO_UI(rmem[17].y)), "r"(__HALF2_TO_UI(rmem[29].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[19].y)), "r"(__HALF2_TO_UI(rmem[31].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[15].y)), "r"(__HALF2_TO_UI(rmem[27].x)), "r"(__HALF2_TO_UI(rmem[23].x)), "r"(__HALF2_TO_UI(rmem[27].y)), "r"(__HALF2_TO_UI(rmem[23].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[19].x)), "r"(__HALF2_TO_UI(rmem[31].x)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<981, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<571>;
.reg .b64 rd<2>;
mov.u32 r559, %tid.y;
shl.b32 r560, r559, 8;
mov.u32 r561, %16;
add.s32 r562, r561, r560;
mov.u32 r563, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f12, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r102, {low, high};
}
mov.f32 f44, 0f3F800000;
mov.f32 f10, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f10;
cvt.rn.f16.f32 high, f10;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r106, {low, high};
}
mov.f32 f43, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
and.b32 r564, r563, 3;
shl.b32 r565, r563, 6;
and.b32 r566, r565, -256;
add.s32 r567, r562, r566;
cvt.rn.f32.u32 f47, r564;
mul.f32 f48, f47, 0f3E490FDB;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
fma.rn.f16x2 r207, r161, r200, r204;
}
{
mul.f16x2 r211, r161, r202;
}
{
neg.f16x2 r214, r211;
}
{
fma.rn.f16x2 r216, r164, r200, r214;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
fma.rn.f16x2 r244, r173, r237, r241;
}
{
mul.f16x2 r248, r173, r239;
}
{
neg.f16x2 r251, r248;
}
{
fma.rn.f16x2 r253, r176, r237, r251;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
fma.rn.f16x2 r281, r185, r274, r278;
}
{
mul.f16x2 r285, r185, r276;
}
{
neg.f16x2 r288, r285;
}
{
fma.rn.f16x2 r290, r188, r274, r288;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
fma.rn.f16x2 r318, r155, r311, r315;
}
{
mul.f16x2 r322, r155, r313;
}
{
neg.f16x2 r325, r322;
}
{
fma.rn.f16x2 r327, r158, r311, r325;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
fma.rn.f16x2 r355, r167, r348, r352;
}
{
mul.f16x2 r359, r167, r350;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r170, r348, r362;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
fma.rn.f16x2 r392, r179, r385, r389;
}
{
mul.f16x2 r396, r179, r387;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r182, r385, r399;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
fma.rn.f16x2 r429, r191, r422, r426;
}
{
mul.f16x2 r433, r191, r424;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r194, r422, r436;
}
barrier.sync 0;
and.b32 r568, r565, 192;
add.s32 r569, r567, r568;
st.shared.v4.f32 [r569], {r149, r152, r207, r216};
st.shared.v4.f32 [r569+16], {r244, r253, r281, r290};
st.shared.v4.f32 [r569+32], {r318, r327, r355, r364};
st.shared.v4.f32 [r569+48], {r392, r401, r429, r438};
barrier.sync 0;
mad.lo.s32 r570, r564, -56, r569;
ld.shared.u32 r460, [r570];
ld.shared.u32 r463, [r570+4];
ld.shared.u32 r510, [r570+32];
ld.shared.u32 r513, [r570+36];
ld.shared.u32 r472, [r570+64];
ld.shared.u32 r475, [r570+68];
ld.shared.u32 r522, [r570+96];
ld.shared.u32 r525, [r570+100];
ld.shared.u32 r461, [r570+128];
ld.shared.u32 r464, [r570+132];
ld.shared.u32 r511, [r570+160];
ld.shared.u32 r514, [r570+164];
ld.shared.u32 r473, [r570+192];
ld.shared.u32 r476, [r570+196];
ld.shared.u32 r523, [r570+224];
ld.shared.u32 r526, [r570+228];
{
add.f16x2 r459, r460, r461;
}
{
add.f16x2 r462, r463, r464;
}
{
sub.f16x2 r465, r460, r461;
}
{
sub.f16x2 r468, r463, r464;
}
{
add.f16x2 r471, r472, r473;
}
{
add.f16x2 r474, r475, r476;
}
{
sub.f16x2 r477, r472, r473;
}
{
sub.f16x2 r480, r475, r476;
}
{
neg.f16x2 r483, r480;
}
{
add.f16x2 %0, r459, r471;
}
{
add.f16x2 %1, r462, r474;
}
{
sub.f16x2 %8, r459, r471;
}
{
sub.f16x2 %9, r462, r474;
}
{
add.f16x2 %4, r465, r483;
}
{
add.f16x2 %5, r468, r477;
}
{
sub.f16x2 %12, r465, r483;
}
{
sub.f16x2 %13, r468, r477;
}
{
add.f16x2 r509, r510, r511;
}
{
add.f16x2 r512, r513, r514;
}
{
sub.f16x2 r515, r510, r511;
}
{
sub.f16x2 r518, r513, r514;
}
{
add.f16x2 r521, r522, r523;
}
{
add.f16x2 r524, r525, r526;
}
{
sub.f16x2 r527, r522, r523;
}
{
sub.f16x2 r530, r525, r526;
}
{
neg.f16x2 r533, r530;
}
{
add.f16x2 %2, r509, r521;
}
{
add.f16x2 %3, r512, r524;
}
{
sub.f16x2 %10, r509, r521;
}
{
sub.f16x2 %11, r512, r524;
}
{
add.f16x2 %6, r515, r533;
}
{
add.f16x2 %7, r518, r527;
}
{
sub.f16x2 %14, r515, r533;
}
{
sub.f16x2 %15, r518, r527;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<982, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<50>;
.reg .b32 r<571>;
.reg .b64 rd<2>;
mov.u32 r559, %tid.y;
shl.b32 r560, r559, 7;
mov.u32 r561, %16;
add.s32 r562, r561, r560;
mov.u32 r563, %tid.x;
{
add.f16x2 r1, %17, %25;
}
{
add.f16x2 r4, %18, %26;
}
{
sub.f16x2 r7, %17, %25;
}
{
sub.f16x2 r10, %18, %26;
}
{
add.f16x2 r13, %21, %29;
}
{
add.f16x2 r16, %22, %30;
}
{
sub.f16x2 r19, %21, %29;
}
{
sub.f16x2 r22, %22, %30;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %19, %27;
}
{
add.f16x2 r54, %20, %28;
}
{
sub.f16x2 r57, %19, %27;
}
{
sub.f16x2 r60, %20, %28;
}
{
add.f16x2 r63, %23, %31;
}
{
add.f16x2 r66, %24, %32;
}
{
sub.f16x2 r69, %23, %31;
}
{
sub.f16x2 r72, %24, %32;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f12, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r102, {low, high};
}
mov.f32 f44, 0f3F800000;
mov.f32 f10, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f10;
cvt.rn.f16.f32 high, f10;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f12;
cvt.rn.f16.f32 high, f12;
mov.b32 r106, {low, high};
}
mov.f32 f43, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
and.b32 r564, r563, 3;
shl.b32 r565, r563, 5;
and.b32 r566, r565, -128;
add.s32 r567, r562, r566;
cvt.rn.f32.u32 f47, r564;
mul.f32 f48, f47, 0f3E490FDB;
cos.approx.f32 f29, f48;
sin.approx.f32 f49, f48;
neg.f32 f30, f49;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f29;
cvt.rn.f16.f32 high, f30;
mov.b32 r197, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r200, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r202, {high, high};
}
{
mul.f16x2 r204, r164, r202;
}
{
fma.rn.f16x2 r207, r161, r200, r204;
}
{
mul.f16x2 r211, r161, r202;
}
{
neg.f16x2 r214, r211;
}
{
fma.rn.f16x2 r216, r164, r200, r214;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r220, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r222, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r224, {low, high};
}
{
mul.f16x2 r225, r222, r224;
}
{
mul.f16x2 r228, r197, r220;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r231, {high, low};
}
{
fma.rn.f16x2 r233, r225, r231, r228;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r237, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r239, {high, high};
}
{
mul.f16x2 r241, r176, r239;
}
{
fma.rn.f16x2 r244, r173, r237, r241;
}
{
mul.f16x2 r248, r173, r239;
}
{
neg.f16x2 r251, r248;
}
{
fma.rn.f16x2 r253, r176, r237, r251;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r257, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r259, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r261, {low, high};
}
{
mul.f16x2 r262, r259, r261;
}
{
mul.f16x2 r265, r233, r257;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r233;
mov.b32 r268, {high, low};
}
{
fma.rn.f16x2 r270, r262, r268, r265;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r274, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r276, {high, high};
}
{
mul.f16x2 r278, r188, r276;
}
{
fma.rn.f16x2 r281, r185, r274, r278;
}
{
mul.f16x2 r285, r185, r276;
}
{
neg.f16x2 r288, r285;
}
{
fma.rn.f16x2 r290, r188, r274, r288;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r294, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r296, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r298, {low, high};
}
{
mul.f16x2 r299, r296, r298;
}
{
mul.f16x2 r302, r270, r294;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r270;
mov.b32 r305, {high, low};
}
{
fma.rn.f16x2 r307, r299, r305, r302;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r311, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r313, {high, high};
}
{
mul.f16x2 r315, r158, r313;
}
{
fma.rn.f16x2 r318, r155, r311, r315;
}
{
mul.f16x2 r322, r155, r313;
}
{
neg.f16x2 r325, r322;
}
{
fma.rn.f16x2 r327, r158, r311, r325;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r331, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r333, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r335, {low, high};
}
{
mul.f16x2 r336, r333, r335;
}
{
mul.f16x2 r339, r307, r331;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r307;
mov.b32 r342, {high, low};
}
{
fma.rn.f16x2 r344, r336, r342, r339;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r348, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r350, {high, high};
}
{
mul.f16x2 r352, r170, r350;
}
{
fma.rn.f16x2 r355, r167, r348, r352;
}
{
mul.f16x2 r359, r167, r350;
}
{
neg.f16x2 r362, r359;
}
{
fma.rn.f16x2 r364, r170, r348, r362;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r368, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r370, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r372, {low, high};
}
{
mul.f16x2 r373, r370, r372;
}
{
mul.f16x2 r376, r344, r368;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r344;
mov.b32 r379, {high, low};
}
{
fma.rn.f16x2 r381, r373, r379, r376;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r385, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r387, {high, high};
}
{
mul.f16x2 r389, r182, r387;
}
{
fma.rn.f16x2 r392, r179, r385, r389;
}
{
mul.f16x2 r396, r179, r387;
}
{
neg.f16x2 r399, r396;
}
{
fma.rn.f16x2 r401, r182, r385, r399;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r405, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r197;
mov.b32 r407, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f43;
cvt.rn.f16.f32 high, f44;
mov.b32 r409, {low, high};
}
{
mul.f16x2 r410, r407, r409;
}
{
mul.f16x2 r413, r381, r405;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r381;
mov.b32 r416, {high, low};
}
{
fma.rn.f16x2 r418, r410, r416, r413;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r422, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r418;
mov.b32 r424, {high, high};
}
{
mul.f16x2 r426, r194, r424;
}
{
fma.rn.f16x2 r429, r191, r422, r426;
}
{
mul.f16x2 r433, r191, r424;
}
{
neg.f16x2 r436, r433;
}
{
fma.rn.f16x2 r438, r194, r422, r436;
}
barrier.sync 0;
and.b32 r568, r565, 96;
add.s32 r569, r567, r568;
st.shared.v4.f32 [r569], {r149, r207, r244, r281};
st.shared.v4.f32 [r569+16], {r318, r355, r392, r429};
barrier.sync 0;
mad.lo.s32 r570, r564, -28, r569;
ld.shared.u32 r460, [r570];
ld.shared.u32 r510, [r570+16];
ld.shared.u32 r472, [r570+32];
ld.shared.u32 r522, [r570+48];
ld.shared.u32 r461, [r570+64];
ld.shared.u32 r511, [r570+80];
ld.shared.u32 r473, [r570+96];
ld.shared.u32 r523, [r570+112];
barrier.sync 0;
st.shared.v4.f32 [r569], {r152, r216, r253, r290};
st.shared.v4.f32 [r569+16], {r327, r364, r401, r438};
barrier.sync 0;
ld.shared.u32 r463, [r570];
ld.shared.u32 r513, [r570+16];
ld.shared.u32 r475, [r570+32];
ld.shared.u32 r525, [r570+48];
ld.shared.u32 r464, [r570+64];
ld.shared.u32 r514, [r570+80];
ld.shared.u32 r476, [r570+96];
ld.shared.u32 r526, [r570+112];
{
add.f16x2 r459, r460, r461;
}
{
add.f16x2 r462, r463, r464;
}
{
sub.f16x2 r465, r460, r461;
}
{
sub.f16x2 r468, r463, r464;
}
{
add.f16x2 r471, r472, r473;
}
{
add.f16x2 r474, r475, r476;
}
{
sub.f16x2 r477, r472, r473;
}
{
sub.f16x2 r480, r475, r476;
}
{
neg.f16x2 r483, r480;
}
{
add.f16x2 %0, r459, r471;
}
{
add.f16x2 %1, r462, r474;
}
{
sub.f16x2 %8, r459, r471;
}
{
sub.f16x2 %9, r462, r474;
}
{
add.f16x2 %4, r465, r483;
}
{
add.f16x2 %5, r468, r477;
}
{
sub.f16x2 %12, r465, r483;
}
{
sub.f16x2 %13, r468, r477;
}
{
add.f16x2 r509, r510, r511;
}
{
add.f16x2 r512, r513, r514;
}
{
sub.f16x2 r515, r510, r511;
}
{
sub.f16x2 r518, r513, r514;
}
{
add.f16x2 r521, r522, r523;
}
{
add.f16x2 r524, r525, r526;
}
{
sub.f16x2 r527, r522, r523;
}
{
sub.f16x2 r530, r525, r526;
}
{
neg.f16x2 r533, r530;
}
{
add.f16x2 %2, r509, r521;
}
{
add.f16x2 %3, r512, r524;
}
{
sub.f16x2 %10, r509, r521;
}
{
sub.f16x2 %11, r512, r524;
}
{
add.f16x2 %6, r515, r533;
}
{
add.f16x2 %7, r518, r527;
}
{
sub.f16x2 %14, r515, r533;
}
{
sub.f16x2 %15, r518, r527;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<983, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<27>;
.reg .b32 r<373>;
.reg .b64 rd<2>;
mov.u32 r353, %tid.y;
shl.b32 r354, r353, 8;
mov.u32 r355, %8;
add.s32 r356, r355, r354;
mov.u32 r357, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
and.b32 r358, r357, 7;
shl.b32 r359, r357, 5;
and.b32 r360, r359, -256;
add.s32 r361, r356, r360;
cvt.rn.f32.u32 f21, r358;
mul.f32 f22, f21, 0f3E490FDB;
cos.approx.f32 f1, f22;
sin.approx.f32 f23, f22;
neg.f32 f2, f23;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
fma.rn.f16x2 r61, r39, r54, r58;
}
{
mul.f16x2 r65, r39, r56;
}
{
neg.f16x2 r68, r65;
}
{
fma.rn.f16x2 r70, r42, r54, r68;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f17, 0fBF800000;
mov.f32 f18, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
fma.rn.f16x2 r98, r33, r91, r95;
}
{
mul.f16x2 r102, r33, r93;
}
{
neg.f16x2 r105, r102;
}
{
fma.rn.f16x2 r107, r36, r91, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
fma.rn.f16x2 r135, r45, r128, r132;
}
{
mul.f16x2 r139, r45, r130;
}
{
neg.f16x2 r142, r139;
}
{
fma.rn.f16x2 r144, r48, r128, r142;
}
barrier.sync 0;
and.b32 r362, r359, 224;
add.s32 r363, r361, r362;
st.shared.v4.f32 [r363], {r27, r30, r61, r70};
st.shared.v4.f32 [r363+16], {r98, r107, r135, r144};
barrier.sync 0;
mad.lo.s32 r364, r358, -24, r363;
ld.shared.u32 r166, [r364];
ld.shared.u32 r169, [r364+4];
ld.shared.u32 r178, [r364+64];
ld.shared.u32 r181, [r364+68];
ld.shared.u32 r167, [r364+128];
ld.shared.u32 r170, [r364+132];
ld.shared.u32 r179, [r364+192];
ld.shared.u32 r182, [r364+196];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r186;
}
{
add.f16x2 r191, r165, r177;
}
{
add.f16x2 r194, r168, r180;
}
{
sub.f16x2 r197, r165, r177;
}
{
sub.f16x2 r200, r168, r180;
}
{
add.f16x2 r203, r171, r189;
}
{
add.f16x2 r206, r174, r183;
}
{
sub.f16x2 r209, r171, r189;
}
{
sub.f16x2 r212, r174, r183;
}
and.b32 r365, r357, 4;
bfe.u32 r366, r357, 2, 1;
cvt.rn.f32.u32 f24, r366;
mul.f32 f25, f24, 0f3F490FDB;
cos.approx.f32 f11, f25;
sin.approx.f32 f26, f25;
neg.f32 f12, f26;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f11;
cvt.rn.f16.f32 high, f12;
mov.b32 r215, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r218, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r220, {high, high};
}
{
mul.f16x2 r222, r206, r220;
}
{
fma.rn.f16x2 r225, r203, r218, r222;
}
{
mul.f16x2 r229, r203, r220;
}
{
neg.f16x2 r232, r229;
}
{
fma.rn.f16x2 r234, r206, r218, r232;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r240, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r242, {low, high};
}
{
mul.f16x2 r243, r240, r242;
}
{
mul.f16x2 r246, r215, r238;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r249, {high, low};
}
{
fma.rn.f16x2 r251, r243, r249, r246;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r255, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r257, {high, high};
}
{
mul.f16x2 r259, r200, r257;
}
{
fma.rn.f16x2 r262, r197, r255, r259;
}
{
mul.f16x2 r266, r197, r257;
}
{
neg.f16x2 r269, r266;
}
{
fma.rn.f16x2 r271, r200, r255, r269;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r277, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r279, {low, high};
}
{
mul.f16x2 r280, r277, r279;
}
{
mul.f16x2 r283, r251, r275;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r286, {high, low};
}
{
fma.rn.f16x2 r288, r280, r286, r283;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r294, {high, high};
}
{
mul.f16x2 r296, r212, r294;
}
{
fma.rn.f16x2 r299, r209, r292, r296;
}
{
mul.f16x2 r303, r209, r294;
}
{
neg.f16x2 r306, r303;
}
{
fma.rn.f16x2 r308, r212, r292, r306;
}
shl.b32 r367, r357, 3;
and.b32 r368, r367, 24;
add.s32 r369, r361, r368;
barrier.sync 0;
and.b32 r370, r359, 128;
add.s32 r371, r369, r370;
st.shared.u32 [r371], r191;
st.shared.u32 [r371+4], r194;
st.shared.u32 [r371+32], r225;
st.shared.u32 [r371+36], r234;
st.shared.u32 [r371+64], r262;
st.shared.u32 [r371+68], r271;
st.shared.u32 [r371+96], r299;
st.shared.u32 [r371+100], r308;
barrier.sync 0;
mad.lo.s32 r372, r365, -24, r371;
ld.shared.u32 r330, [r372];
ld.shared.u32 r333, [r372+4];
ld.shared.u32 r342, [r372+64];
ld.shared.u32 r345, [r372+68];
ld.shared.u32 r331, [r372+128];
ld.shared.u32 r334, [r372+132];
ld.shared.u32 r343, [r372+192];
ld.shared.u32 r346, [r372+196];
{
add.f16x2 %0, r330, r331;
}
{
add.f16x2 %1, r333, r334;
}
{
sub.f16x2 %4, r330, r331;
}
{
sub.f16x2 %5, r333, r334;
}
{
add.f16x2 %2, r342, r343;
}
{
add.f16x2 %3, r345, r346;
}
{
sub.f16x2 %6, r342, r343;
}
{
sub.f16x2 %7, r345, r346;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<984, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<27>;
.reg .b32 r<373>;
.reg .b64 rd<2>;
mov.u32 r353, %tid.y;
shl.b32 r354, r353, 7;
mov.u32 r355, %8;
add.s32 r356, r355, r354;
mov.u32 r357, %tid.x;
{
add.f16x2 r1, %9, %13;
}
{
add.f16x2 r4, %10, %14;
}
{
sub.f16x2 r7, %9, %13;
}
{
sub.f16x2 r10, %10, %14;
}
{
add.f16x2 r13, %11, %15;
}
{
add.f16x2 r16, %12, %16;
}
{
sub.f16x2 r19, %11, %15;
}
{
sub.f16x2 r22, %12, %16;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
and.b32 r358, r357, 7;
shl.b32 r359, r357, 4;
and.b32 r360, r359, -128;
add.s32 r361, r356, r360;
cvt.rn.f32.u32 f21, r358;
mul.f32 f22, f21, 0f3E490FDB;
cos.approx.f32 f1, f22;
sin.approx.f32 f23, f22;
neg.f32 f2, f23;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r51, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r54, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r56, {high, high};
}
{
mul.f16x2 r58, r42, r56;
}
{
fma.rn.f16x2 r61, r39, r54, r58;
}
{
mul.f16x2 r65, r39, r56;
}
{
neg.f16x2 r68, r65;
}
{
fma.rn.f16x2 r70, r42, r54, r68;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r74, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r76, {high, high};
}
mov.f32 f17, 0fBF800000;
mov.f32 f18, 0f3F800000;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r78, {low, high};
}
{
mul.f16x2 r79, r76, r78;
}
{
mul.f16x2 r82, r51, r74;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r85, {high, low};
}
{
fma.rn.f16x2 r87, r79, r85, r82;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r91, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r93, {high, high};
}
{
mul.f16x2 r95, r36, r93;
}
{
fma.rn.f16x2 r98, r33, r91, r95;
}
{
mul.f16x2 r102, r33, r93;
}
{
neg.f16x2 r105, r102;
}
{
fma.rn.f16x2 r107, r36, r91, r105;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r111, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r51;
mov.b32 r113, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r115, {low, high};
}
{
mul.f16x2 r116, r113, r115;
}
{
mul.f16x2 r119, r87, r111;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r87;
mov.b32 r122, {high, low};
}
{
fma.rn.f16x2 r124, r116, r122, r119;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r128, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r124;
mov.b32 r130, {high, high};
}
{
mul.f16x2 r132, r48, r130;
}
{
fma.rn.f16x2 r135, r45, r128, r132;
}
{
mul.f16x2 r139, r45, r130;
}
{
neg.f16x2 r142, r139;
}
{
fma.rn.f16x2 r144, r48, r128, r142;
}
barrier.sync 0;
and.b32 r362, r359, 112;
add.s32 r363, r361, r362;
st.shared.v4.f32 [r363], {r27, r61, r98, r135};
barrier.sync 0;
mad.lo.s32 r364, r358, -12, r363;
ld.shared.u32 r166, [r364];
ld.shared.u32 r178, [r364+32];
ld.shared.u32 r167, [r364+64];
ld.shared.u32 r179, [r364+96];
barrier.sync 0;
st.shared.v4.f32 [r363], {r30, r70, r107, r144};
barrier.sync 0;
ld.shared.u32 r169, [r364];
ld.shared.u32 r181, [r364+32];
ld.shared.u32 r170, [r364+64];
ld.shared.u32 r182, [r364+96];
{
add.f16x2 r165, r166, r167;
}
{
add.f16x2 r168, r169, r170;
}
{
sub.f16x2 r171, r166, r167;
}
{
sub.f16x2 r174, r169, r170;
}
{
add.f16x2 r177, r178, r179;
}
{
add.f16x2 r180, r181, r182;
}
{
sub.f16x2 r183, r178, r179;
}
{
sub.f16x2 r186, r181, r182;
}
{
neg.f16x2 r189, r186;
}
{
add.f16x2 r191, r165, r177;
}
{
add.f16x2 r194, r168, r180;
}
{
sub.f16x2 r197, r165, r177;
}
{
sub.f16x2 r200, r168, r180;
}
{
add.f16x2 r203, r171, r189;
}
{
add.f16x2 r206, r174, r183;
}
{
sub.f16x2 r209, r171, r189;
}
{
sub.f16x2 r212, r174, r183;
}
and.b32 r365, r357, 4;
bfe.u32 r366, r357, 2, 1;
shl.b32 r367, r357, 2;
and.b32 r368, r367, 12;
add.s32 r369, r361, r368;
cvt.rn.f32.u32 f24, r366;
mul.f32 f25, f24, 0f3F490FDB;
cos.approx.f32 f11, f25;
sin.approx.f32 f26, f25;
neg.f32 f12, f26;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f11;
cvt.rn.f16.f32 high, f12;
mov.b32 r215, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r218, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r220, {high, high};
}
{
mul.f16x2 r222, r206, r220;
}
{
fma.rn.f16x2 r225, r203, r218, r222;
}
{
mul.f16x2 r229, r203, r220;
}
{
neg.f16x2 r232, r229;
}
{
fma.rn.f16x2 r234, r206, r218, r232;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r238, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r240, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r242, {low, high};
}
{
mul.f16x2 r243, r240, r242;
}
{
mul.f16x2 r246, r215, r238;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r249, {high, low};
}
{
fma.rn.f16x2 r251, r243, r249, r246;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r255, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r257, {high, high};
}
{
mul.f16x2 r259, r200, r257;
}
{
fma.rn.f16x2 r262, r197, r255, r259;
}
{
mul.f16x2 r266, r197, r257;
}
{
neg.f16x2 r269, r266;
}
{
fma.rn.f16x2 r271, r200, r255, r269;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r275, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r215;
mov.b32 r277, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f17;
cvt.rn.f16.f32 high, f18;
mov.b32 r279, {low, high};
}
{
mul.f16x2 r280, r277, r279;
}
{
mul.f16x2 r283, r251, r275;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r251;
mov.b32 r286, {high, low};
}
{
fma.rn.f16x2 r288, r280, r286, r283;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r292, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r288;
mov.b32 r294, {high, high};
}
{
mul.f16x2 r296, r212, r294;
}
{
fma.rn.f16x2 r299, r209, r292, r296;
}
{
mul.f16x2 r303, r209, r294;
}
{
neg.f16x2 r306, r303;
}
{
fma.rn.f16x2 r308, r212, r292, r306;
}
barrier.sync 0;
and.b32 r370, r359, 64;
add.s32 r371, r369, r370;
st.shared.u32 [r371], r191;
st.shared.u32 [r371+16], r225;
st.shared.u32 [r371+32], r262;
st.shared.u32 [r371+48], r299;
barrier.sync 0;
mad.lo.s32 r372, r365, -12, r371;
ld.shared.u32 r330, [r372];
ld.shared.u32 r342, [r372+32];
ld.shared.u32 r331, [r372+64];
ld.shared.u32 r343, [r372+96];
barrier.sync 0;
st.shared.u32 [r371], r194;
st.shared.u32 [r371+16], r234;
st.shared.u32 [r371+32], r271;
st.shared.u32 [r371+48], r308;
barrier.sync 0;
ld.shared.u32 r333, [r372];
ld.shared.u32 r345, [r372+32];
ld.shared.u32 r334, [r372+64];
ld.shared.u32 r346, [r372+96];
{
add.f16x2 %0, r330, r331;
}
{
add.f16x2 %1, r333, r334;
}
{
sub.f16x2 %4, r330, r331;
}
{
sub.f16x2 %5, r333, r334;
}
{
add.f16x2 %2, r342, r343;
}
{
add.f16x2 %3, r345, r346;
}
{
sub.f16x2 %6, r342, r343;
}
{
sub.f16x2 %7, r345, r346;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<985, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<37>;
.reg .b32 r<256>;
.reg .b64 rd<2>;
mov.u32 r221, %tid.y;
shl.b32 r222, r221, 8;
mov.u32 r223, %4;
add.s32 r224, r223, r222;
mov.u32 r225, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r226, r225, 15;
shl.b32 r227, r225, 4;
and.b32 r228, r227, -256;
add.s32 r229, r224, r228;
cvt.rn.f32.u32 f25, r226;
mul.f32 f26, f25, 0f3E490FDB;
cos.approx.f32 f1, f26;
sin.approx.f32 f27, f26;
neg.f32 f2, f27;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r230, r227, 240;
add.s32 r231, r229, r230;
st.shared.v2.f32 [r231], {r1, r4};
st.shared.v2.f32 [r231+8], {r23, r32};
barrier.sync 0;
shl.b32 r232, r225, 3;
and.b32 r233, r232, 120;
sub.s32 r234, r231, r233;
ld.shared.u32 r54, [r234];
ld.shared.u32 r57, [r234+4];
ld.shared.u32 r55, [r234+128];
ld.shared.u32 r58, [r234+132];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r235, r225, 1, 3;
cvt.rn.f32.u32 f28, r235;
mul.f32 f29, f28, 0f3EC90FDB;
cos.approx.f32 f7, f29;
sin.approx.f32 f30, f29;
neg.f32 f8, f30;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
fma.rn.f16x2 r75, r59, r68, r72;
}
{
mul.f16x2 r79, r59, r70;
}
{
neg.f16x2 r82, r79;
}
{
fma.rn.f16x2 r84, r62, r68, r82;
}
and.b32 r236, r232, 8;
add.s32 r237, r229, r236;
barrier.sync 0;
and.b32 r238, r227, 224;
add.s32 r239, r237, r238;
st.shared.u32 [r239], r53;
st.shared.u32 [r239+4], r56;
st.shared.u32 [r239+16], r75;
st.shared.u32 [r239+20], r84;
barrier.sync 0;
and.b32 r240, r232, 112;
sub.s32 r241, r239, r240;
ld.shared.u32 r106, [r241];
ld.shared.u32 r109, [r241+4];
ld.shared.u32 r107, [r241+128];
ld.shared.u32 r110, [r241+132];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r242, r225, 2, 2;
cvt.rn.f32.u32 f31, r242;
mul.f32 f32, f31, 0f3F490FDB;
cos.approx.f32 f13, f32;
sin.approx.f32 f33, f32;
neg.f32 f14, f33;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
fma.rn.f16x2 r127, r111, r120, r124;
}
{
mul.f16x2 r131, r111, r122;
}
{
neg.f16x2 r134, r131;
}
{
fma.rn.f16x2 r136, r114, r120, r134;
}
and.b32 r243, r232, 24;
add.s32 r244, r229, r243;
barrier.sync 0;
and.b32 r245, r227, 192;
add.s32 r246, r244, r245;
st.shared.u32 [r246], r105;
st.shared.u32 [r246+4], r108;
st.shared.u32 [r246+32], r127;
st.shared.u32 [r246+36], r136;
barrier.sync 0;
and.b32 r247, r232, 96;
sub.s32 r248, r246, r247;
ld.shared.u32 r158, [r248];
ld.shared.u32 r161, [r248+4];
ld.shared.u32 r159, [r248+128];
ld.shared.u32 r162, [r248+132];
{
add.f16x2 r157, r158, r159;
}
{
add.f16x2 r160, r161, r162;
}
{
sub.f16x2 r163, r158, r159;
}
{
sub.f16x2 r166, r161, r162;
}
bfe.u32 r249, r225, 3, 1;
cvt.rn.f32.u32 f34, r249;
mul.f32 f35, f34, 0f3FC90FDB;
cos.approx.f32 f19, f35;
sin.approx.f32 f36, f35;
neg.f32 f20, f36;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f19;
cvt.rn.f16.f32 high, f20;
mov.b32 r169, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r172, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r174, {high, high};
}
{
mul.f16x2 r176, r166, r174;
}
{
fma.rn.f16x2 r179, r163, r172, r176;
}
{
mul.f16x2 r183, r163, r174;
}
{
neg.f16x2 r186, r183;
}
{
fma.rn.f16x2 r188, r166, r172, r186;
}
and.b32 r250, r232, 56;
add.s32 r251, r229, r250;
barrier.sync 0;
and.b32 r252, r227, 128;
add.s32 r253, r251, r252;
st.shared.u32 [r253], r157;
st.shared.u32 [r253+4], r160;
st.shared.u32 [r253+64], r179;
st.shared.u32 [r253+68], r188;
barrier.sync 0;
and.b32 r254, r232, 64;
sub.s32 r255, r253, r254;
ld.shared.u32 r210, [r255];
ld.shared.u32 r213, [r255+4];
ld.shared.u32 r211, [r255+128];
ld.shared.u32 r214, [r255+132];
{
add.f16x2 %0, r210, r211;
}
{
add.f16x2 %1, r213, r214;
}
{
sub.f16x2 %2, r210, r211;
}
{
sub.f16x2 %3, r213, r214;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<986, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<154>;
.reg .b32 r<1283>;
.reg .b64 rd<2>;
mov.u32 r1271, %tid.y;
shl.b32 r1272, r1271, 8;
mov.u32 r1273, %32;
add.s32 r1274, r1273, r1272;
mov.u32 r1275, %tid.x;
{
add.f16x2 r1, %33, %49;
}
{
add.f16x2 r4, %34, %50;
}
{
sub.f16x2 r7, %33, %49;
}
{
sub.f16x2 r10, %34, %50;
}
{
add.f16x2 r13, %41, %57;
}
{
add.f16x2 r16, %42, %58;
}
{
sub.f16x2 r19, %41, %57;
}
{
sub.f16x2 r22, %42, %58;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %37, %53;
}
{
add.f16x2 r54, %38, %54;
}
{
sub.f16x2 r57, %37, %53;
}
{
sub.f16x2 r60, %38, %54;
}
{
add.f16x2 r63, %45, %61;
}
{
add.f16x2 r66, %46, %62;
}
{
sub.f16x2 r69, %45, %61;
}
{
sub.f16x2 r72, %46, %62;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f80, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r102, {low, high};
}
mov.f32 f148, 0f3F800000;
mov.f32 f78, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r106, {low, high};
}
mov.f32 f147, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %35, %51;
}
{
add.f16x2 r200, %36, %52;
}
{
sub.f16x2 r203, %35, %51;
}
{
sub.f16x2 r206, %36, %52;
}
{
add.f16x2 r209, %43, %59;
}
{
add.f16x2 r212, %44, %60;
}
{
sub.f16x2 r215, %43, %59;
}
{
sub.f16x2 r218, %44, %60;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %39, %55;
}
{
add.f16x2 r250, %40, %56;
}
{
sub.f16x2 r253, %39, %55;
}
{
sub.f16x2 r256, %40, %56;
}
{
add.f16x2 r259, %47, %63;
}
{
add.f16x2 r262, %48, %64;
}
{
sub.f16x2 r265, %47, %63;
}
{
sub.f16x2 r268, %48, %64;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f76, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r393, {low, high};
}
mov.f32 f84, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r398, {low, high};
}
mov.f32 f74, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r404, {low, high};
}
mov.f32 f82, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
and.b32 r1276, r1275, 1;
shl.b32 r1277, r1275, 7;
and.b32 r1278, r1277, -256;
add.s32 r1279, r1274, r1278;
cvt.rn.f32.u32 f151, r1276;
mul.f32 f152, f151, 0f3E490FDB;
cos.approx.f32 f117, f152;
sin.approx.f32 f153, f152;
neg.f32 f118, f153;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f117;
cvt.rn.f16.f32 high, f118;
mov.b32 r617, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r536, r622;
}
{
fma.rn.f16x2 r627, r533, r620, r624;
}
{
mul.f16x2 r631, r533, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r536, r620, r634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r640, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r642, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r644, {low, high};
}
{
mul.f16x2 r645, r642, r644;
}
{
mul.f16x2 r648, r617, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r651, {high, low};
}
{
fma.rn.f16x2 r653, r645, r651, r648;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r659, {high, high};
}
{
mul.f16x2 r661, r548, r659;
}
{
fma.rn.f16x2 r664, r545, r657, r661;
}
{
mul.f16x2 r668, r545, r659;
}
{
neg.f16x2 r671, r668;
}
{
fma.rn.f16x2 r673, r548, r657, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r677, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r679, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r679, r681;
}
{
mul.f16x2 r685, r653, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r688, {high, low};
}
{
fma.rn.f16x2 r690, r682, r688, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r696, {high, high};
}
{
mul.f16x2 r698, r560, r696;
}
{
fma.rn.f16x2 r701, r557, r694, r698;
}
{
mul.f16x2 r705, r557, r696;
}
{
neg.f16x2 r708, r705;
}
{
fma.rn.f16x2 r710, r560, r694, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r714, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r716, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r718, {low, high};
}
{
mul.f16x2 r719, r716, r718;
}
{
mul.f16x2 r722, r690, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r725, {high, low};
}
{
fma.rn.f16x2 r727, r719, r725, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r733, {high, high};
}
{
mul.f16x2 r735, r572, r733;
}
{
fma.rn.f16x2 r738, r569, r731, r735;
}
{
mul.f16x2 r742, r569, r733;
}
{
neg.f16x2 r745, r742;
}
{
fma.rn.f16x2 r747, r572, r731, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r751, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r753, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r755, {low, high};
}
{
mul.f16x2 r756, r753, r755;
}
{
mul.f16x2 r759, r727, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r762, {high, low};
}
{
fma.rn.f16x2 r764, r756, r762, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r770, {high, high};
}
{
mul.f16x2 r772, r584, r770;
}
{
fma.rn.f16x2 r775, r581, r768, r772;
}
{
mul.f16x2 r779, r581, r770;
}
{
neg.f16x2 r782, r779;
}
{
fma.rn.f16x2 r784, r584, r768, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r788, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r790, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r792, {low, high};
}
{
mul.f16x2 r793, r790, r792;
}
{
mul.f16x2 r796, r764, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r799, {high, low};
}
{
fma.rn.f16x2 r801, r793, r799, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r807, {high, high};
}
{
mul.f16x2 r809, r596, r807;
}
{
fma.rn.f16x2 r812, r593, r805, r809;
}
{
mul.f16x2 r816, r593, r807;
}
{
neg.f16x2 r819, r816;
}
{
fma.rn.f16x2 r821, r596, r805, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r825, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r827, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r829, {low, high};
}
{
mul.f16x2 r830, r827, r829;
}
{
mul.f16x2 r833, r801, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r836, {high, low};
}
{
fma.rn.f16x2 r838, r830, r836, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r844, {high, high};
}
{
mul.f16x2 r846, r608, r844;
}
{
fma.rn.f16x2 r849, r605, r842, r846;
}
{
mul.f16x2 r853, r605, r844;
}
{
neg.f16x2 r856, r853;
}
{
fma.rn.f16x2 r858, r608, r842, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r862, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r864, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r864, r866;
}
{
mul.f16x2 r870, r838, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r873, {high, low};
}
{
fma.rn.f16x2 r875, r867, r873, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r879, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r881, {high, high};
}
{
mul.f16x2 r883, r530, r881;
}
{
fma.rn.f16x2 r886, r527, r879, r883;
}
{
mul.f16x2 r890, r527, r881;
}
{
neg.f16x2 r893, r890;
}
{
fma.rn.f16x2 r895, r530, r879, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r899, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r901, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r903, {low, high};
}
{
mul.f16x2 r904, r901, r903;
}
{
mul.f16x2 r907, r875, r899;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r910, {high, low};
}
{
fma.rn.f16x2 r912, r904, r910, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r916, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r918, {high, high};
}
{
mul.f16x2 r920, r542, r918;
}
{
fma.rn.f16x2 r923, r539, r916, r920;
}
{
mul.f16x2 r927, r539, r918;
}
{
neg.f16x2 r930, r927;
}
{
fma.rn.f16x2 r932, r542, r916, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r936, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r938, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r940, {low, high};
}
{
mul.f16x2 r941, r938, r940;
}
{
mul.f16x2 r944, r912, r936;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r947, {high, low};
}
{
fma.rn.f16x2 r949, r941, r947, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r953, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r955, {high, high};
}
{
mul.f16x2 r957, r554, r955;
}
{
fma.rn.f16x2 r960, r551, r953, r957;
}
{
mul.f16x2 r964, r551, r955;
}
{
neg.f16x2 r967, r964;
}
{
fma.rn.f16x2 r969, r554, r953, r967;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r973, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r975, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r977, {low, high};
}
{
mul.f16x2 r978, r975, r977;
}
{
mul.f16x2 r981, r949, r973;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r984, {high, low};
}
{
fma.rn.f16x2 r986, r978, r984, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r990, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r992, {high, high};
}
{
mul.f16x2 r994, r566, r992;
}
{
fma.rn.f16x2 r997, r563, r990, r994;
}
{
mul.f16x2 r1001, r563, r992;
}
{
neg.f16x2 r1004, r1001;
}
{
fma.rn.f16x2 r1006, r566, r990, r1004;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1010, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1012, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1014, {low, high};
}
{
mul.f16x2 r1015, r1012, r1014;
}
{
mul.f16x2 r1018, r986, r1010;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r1021, {high, low};
}
{
fma.rn.f16x2 r1023, r1015, r1021, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1027, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1029, {high, high};
}
{
mul.f16x2 r1031, r578, r1029;
}
{
fma.rn.f16x2 r1034, r575, r1027, r1031;
}
{
mul.f16x2 r1038, r575, r1029;
}
{
neg.f16x2 r1041, r1038;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1041;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1047, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1049, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1051, {low, high};
}
{
mul.f16x2 r1052, r1049, r1051;
}
{
mul.f16x2 r1055, r1023, r1047;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1058, {high, low};
}
{
fma.rn.f16x2 r1060, r1052, r1058, r1055;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1064, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1066, {high, high};
}
{
mul.f16x2 r1068, r590, r1066;
}
{
fma.rn.f16x2 r1071, r587, r1064, r1068;
}
{
mul.f16x2 r1075, r587, r1066;
}
{
neg.f16x2 r1078, r1075;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1078;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1084, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1086, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1088, {low, high};
}
{
mul.f16x2 r1089, r1086, r1088;
}
{
mul.f16x2 r1092, r1060, r1084;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1095, {high, low};
}
{
fma.rn.f16x2 r1097, r1089, r1095, r1092;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1103, {high, high};
}
{
mul.f16x2 r1105, r602, r1103;
}
{
fma.rn.f16x2 r1108, r599, r1101, r1105;
}
{
mul.f16x2 r1112, r599, r1103;
}
{
neg.f16x2 r1115, r1112;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1115;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1125, {low, high};
}
{
mul.f16x2 r1126, r1123, r1125;
}
{
mul.f16x2 r1129, r1097, r1121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1132, {high, low};
}
{
fma.rn.f16x2 r1134, r1126, r1132, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1140, {high, high};
}
{
mul.f16x2 r1142, r614, r1140;
}
{
fma.rn.f16x2 r1145, r611, r1138, r1142;
}
{
mul.f16x2 r1149, r611, r1140;
}
{
neg.f16x2 r1152, r1149;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1152;
}
barrier.sync 0;
and.b32 r1280, r1277, 128;
add.s32 r1281, r1279, r1280;
st.shared.v4.f32 [r1281], {r521, r524, r627, r636};
st.shared.v4.f32 [r1281+16], {r664, r673, r701, r710};
st.shared.v4.f32 [r1281+32], {r738, r747, r775, r784};
st.shared.v4.f32 [r1281+48], {r812, r821, r849, r858};
st.shared.v4.f32 [r1281+64], {r886, r895, r923, r932};
st.shared.v4.f32 [r1281+80], {r960, r969, r997, r1006};
st.shared.v4.f32 [r1281+96], {r1034, r1043, r1071, r1080};
st.shared.v4.f32 [r1281+112], {r1108, r1117, r1145, r1154};
barrier.sync 0;
mad.lo.s32 r1282, r1276, -120, r1281;
ld.shared.u32 r1176, [r1282];
ld.shared.u32 r1179, [r1282+4];
ld.shared.u32 r1188, [r1282+16];
ld.shared.u32 r1191, [r1282+20];
ld.shared.u32 r1200, [r1282+32];
ld.shared.u32 r1203, [r1282+36];
ld.shared.u32 r1212, [r1282+48];
ld.shared.u32 r1215, [r1282+52];
ld.shared.u32 r1224, [r1282+64];
ld.shared.u32 r1227, [r1282+68];
ld.shared.u32 r1236, [r1282+80];
ld.shared.u32 r1239, [r1282+84];
ld.shared.u32 r1248, [r1282+96];
ld.shared.u32 r1251, [r1282+100];
ld.shared.u32 r1260, [r1282+112];
ld.shared.u32 r1263, [r1282+116];
ld.shared.u32 r1177, [r1282+128];
ld.shared.u32 r1180, [r1282+132];
ld.shared.u32 r1189, [r1282+144];
ld.shared.u32 r1192, [r1282+148];
ld.shared.u32 r1201, [r1282+160];
ld.shared.u32 r1204, [r1282+164];
ld.shared.u32 r1213, [r1282+176];
ld.shared.u32 r1216, [r1282+180];
ld.shared.u32 r1225, [r1282+192];
ld.shared.u32 r1228, [r1282+196];
ld.shared.u32 r1237, [r1282+208];
ld.shared.u32 r1240, [r1282+212];
ld.shared.u32 r1249, [r1282+224];
ld.shared.u32 r1252, [r1282+228];
ld.shared.u32 r1261, [r1282+240];
ld.shared.u32 r1264, [r1282+244];
{
add.f16x2 %0, r1176, r1177;
}
{
add.f16x2 %1, r1179, r1180;
}
{
sub.f16x2 %16, r1176, r1177;
}
{
sub.f16x2 %17, r1179, r1180;
}
{
add.f16x2 %2, r1188, r1189;
}
{
add.f16x2 %3, r1191, r1192;
}
{
sub.f16x2 %18, r1188, r1189;
}
{
sub.f16x2 %19, r1191, r1192;
}
{
add.f16x2 %4, r1200, r1201;
}
{
add.f16x2 %5, r1203, r1204;
}
{
sub.f16x2 %20, r1200, r1201;
}
{
sub.f16x2 %21, r1203, r1204;
}
{
add.f16x2 %6, r1212, r1213;
}
{
add.f16x2 %7, r1215, r1216;
}
{
sub.f16x2 %22, r1212, r1213;
}
{
sub.f16x2 %23, r1215, r1216;
}
{
add.f16x2 %8, r1224, r1225;
}
{
add.f16x2 %9, r1227, r1228;
}
{
sub.f16x2 %24, r1224, r1225;
}
{
sub.f16x2 %25, r1227, r1228;
}
{
add.f16x2 %10, r1236, r1237;
}
{
add.f16x2 %11, r1239, r1240;
}
{
sub.f16x2 %26, r1236, r1237;
}
{
sub.f16x2 %27, r1239, r1240;
}
{
add.f16x2 %12, r1248, r1249;
}
{
add.f16x2 %13, r1251, r1252;
}
{
sub.f16x2 %28, r1248, r1249;
}
{
sub.f16x2 %29, r1251, r1252;
}
{
add.f16x2 %14, r1260, r1261;
}
{
add.f16x2 %15, r1263, r1264;
}
{
sub.f16x2 %30, r1260, r1261;
}
{
sub.f16x2 %31, r1263, r1264;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<987, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<37>;
.reg .b32 r<256>;
.reg .b64 rd<2>;
mov.u32 r221, %tid.y;
shl.b32 r222, r221, 7;
mov.u32 r223, %4;
add.s32 r224, r223, r222;
mov.u32 r225, %tid.x;
{
add.f16x2 r1, %5, %7;
}
{
add.f16x2 r4, %6, %8;
}
{
sub.f16x2 r7, %5, %7;
}
{
sub.f16x2 r10, %6, %8;
}
and.b32 r226, r225, 15;
shl.b32 r227, r225, 3;
and.b32 r228, r227, -128;
add.s32 r229, r224, r228;
cvt.rn.f32.u32 f25, r226;
mul.f32 f26, f25, 0f3E490FDB;
cos.approx.f32 f1, f26;
sin.approx.f32 f27, f26;
neg.f32 f2, f27;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f1;
cvt.rn.f16.f32 high, f2;
mov.b32 r13, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r16, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r13;
mov.b32 r18, {high, high};
}
{
mul.f16x2 r20, r10, r18;
}
{
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r230, r227, 120;
add.s32 r231, r229, r230;
st.shared.v2.f32 [r231], {r1, r23};
barrier.sync 0;
shl.b32 r232, r225, 2;
and.b32 r233, r232, 60;
sub.s32 r234, r231, r233;
ld.shared.u32 r54, [r234];
ld.shared.u32 r55, [r234+64];
barrier.sync 0;
st.shared.v2.f32 [r231], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r234];
ld.shared.u32 r58, [r234+64];
{
add.f16x2 r53, r54, r55;
}
{
add.f16x2 r56, r57, r58;
}
{
sub.f16x2 r59, r54, r55;
}
{
sub.f16x2 r62, r57, r58;
}
bfe.u32 r235, r225, 1, 3;
and.b32 r236, r232, 4;
add.s32 r237, r229, r236;
cvt.rn.f32.u32 f28, r235;
mul.f32 f29, f28, 0f3EC90FDB;
cos.approx.f32 f7, f29;
sin.approx.f32 f30, f29;
neg.f32 f8, f30;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f7;
cvt.rn.f16.f32 high, f8;
mov.b32 r65, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r68, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r65;
mov.b32 r70, {high, high};
}
{
mul.f16x2 r72, r62, r70;
}
{
fma.rn.f16x2 r75, r59, r68, r72;
}
{
mul.f16x2 r79, r59, r70;
}
{
neg.f16x2 r82, r79;
}
{
fma.rn.f16x2 r84, r62, r68, r82;
}
barrier.sync 0;
and.b32 r238, r227, 112;
add.s32 r239, r237, r238;
st.shared.u32 [r239], r53;
st.shared.u32 [r239+8], r75;
barrier.sync 0;
and.b32 r240, r232, 56;
sub.s32 r241, r239, r240;
ld.shared.u32 r106, [r241];
ld.shared.u32 r107, [r241+64];
barrier.sync 0;
st.shared.u32 [r239], r56;
st.shared.u32 [r239+8], r84;
barrier.sync 0;
ld.shared.u32 r109, [r241];
ld.shared.u32 r110, [r241+64];
{
add.f16x2 r105, r106, r107;
}
{
add.f16x2 r108, r109, r110;
}
{
sub.f16x2 r111, r106, r107;
}
{
sub.f16x2 r114, r109, r110;
}
bfe.u32 r242, r225, 2, 2;
and.b32 r243, r232, 12;
add.s32 r244, r229, r243;
cvt.rn.f32.u32 f31, r242;
mul.f32 f32, f31, 0f3F490FDB;
cos.approx.f32 f13, f32;
sin.approx.f32 f33, f32;
neg.f32 f14, f33;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f13;
cvt.rn.f16.f32 high, f14;
mov.b32 r117, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r120, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r117;
mov.b32 r122, {high, high};
}
{
mul.f16x2 r124, r114, r122;
}
{
fma.rn.f16x2 r127, r111, r120, r124;
}
{
mul.f16x2 r131, r111, r122;
}
{
neg.f16x2 r134, r131;
}
{
fma.rn.f16x2 r136, r114, r120, r134;
}
barrier.sync 0;
and.b32 r245, r227, 96;
add.s32 r246, r244, r245;
st.shared.u32 [r246], r105;
st.shared.u32 [r246+16], r127;
barrier.sync 0;
and.b32 r247, r232, 48;
sub.s32 r248, r246, r247;
ld.shared.u32 r158, [r248];
ld.shared.u32 r159, [r248+64];
barrier.sync 0;
st.shared.u32 [r246], r108;
st.shared.u32 [r246+16], r136;
barrier.sync 0;
ld.shared.u32 r161, [r248];
ld.shared.u32 r162, [r248+64];
{
add.f16x2 r157, r158, r159;
}
{
add.f16x2 r160, r161, r162;
}
{
sub.f16x2 r163, r158, r159;
}
{
sub.f16x2 r166, r161, r162;
}
bfe.u32 r249, r225, 3, 1;
and.b32 r250, r232, 28;
add.s32 r251, r229, r250;
cvt.rn.f32.u32 f34, r249;
mul.f32 f35, f34, 0f3FC90FDB;
cos.approx.f32 f19, f35;
sin.approx.f32 f36, f35;
neg.f32 f20, f36;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f19;
cvt.rn.f16.f32 high, f20;
mov.b32 r169, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r172, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r169;
mov.b32 r174, {high, high};
}
{
mul.f16x2 r176, r166, r174;
}
{
fma.rn.f16x2 r179, r163, r172, r176;
}
{
mul.f16x2 r183, r163, r174;
}
{
neg.f16x2 r186, r183;
}
{
fma.rn.f16x2 r188, r166, r172, r186;
}
barrier.sync 0;
and.b32 r252, r227, 64;
add.s32 r253, r251, r252;
st.shared.u32 [r253], r157;
st.shared.u32 [r253+32], r179;
barrier.sync 0;
and.b32 r254, r232, 32;
sub.s32 r255, r253, r254;
ld.shared.u32 r210, [r255];
ld.shared.u32 r211, [r255+64];
barrier.sync 0;
st.shared.u32 [r253], r160;
st.shared.u32 [r253+32], r188;
barrier.sync 0;
ld.shared.u32 r213, [r255];
ld.shared.u32 r214, [r255+64];
{
add.f16x2 %0, r210, r211;
}
{
add.f16x2 %1, r213, r214;
}
{
sub.f16x2 %2, r210, r211;
}
{
sub.f16x2 %3, r213, r214;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<988, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<154>;
.reg .b32 r<1283>;
.reg .b64 rd<2>;
mov.u32 r1271, %tid.y;
shl.b32 r1272, r1271, 7;
mov.u32 r1273, %32;
add.s32 r1274, r1273, r1272;
mov.u32 r1275, %tid.x;
{
add.f16x2 r1, %33, %49;
}
{
add.f16x2 r4, %34, %50;
}
{
sub.f16x2 r7, %33, %49;
}
{
sub.f16x2 r10, %34, %50;
}
{
add.f16x2 r13, %41, %57;
}
{
add.f16x2 r16, %42, %58;
}
{
sub.f16x2 r19, %41, %57;
}
{
sub.f16x2 r22, %42, %58;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 r27, r1, r13;
}
{
add.f16x2 r30, r4, r16;
}
{
sub.f16x2 r33, r1, r13;
}
{
sub.f16x2 r36, r4, r16;
}
{
add.f16x2 r39, r7, r25;
}
{
add.f16x2 r42, r10, r19;
}
{
sub.f16x2 r45, r7, r25;
}
{
sub.f16x2 r48, r10, r19;
}
{
add.f16x2 r51, %37, %53;
}
{
add.f16x2 r54, %38, %54;
}
{
sub.f16x2 r57, %37, %53;
}
{
sub.f16x2 r60, %38, %54;
}
{
add.f16x2 r63, %45, %61;
}
{
add.f16x2 r66, %46, %62;
}
{
sub.f16x2 r69, %45, %61;
}
{
sub.f16x2 r72, %46, %62;
}
{
neg.f16x2 r75, r72;
}
{
add.f16x2 r77, r51, r63;
}
{
add.f16x2 r80, r54, r66;
}
{
sub.f16x2 r83, r51, r63;
}
{
sub.f16x2 r86, r54, r66;
}
{
add.f16x2 r89, r57, r75;
}
{
add.f16x2 r92, r60, r69;
}
{
sub.f16x2 r95, r57, r75;
}
{
sub.f16x2 r98, r60, r69;
}
mov.f32 f80, 0f3F3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r101, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r102, {low, high};
}
mov.f32 f148, 0f3F800000;
mov.f32 f78, 0fBF3504F3;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r105, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r106, {low, high};
}
mov.f32 f147, 0fBF800000;
{
mul.f16x2 r115, r89, r101;
}
{
mul.f16x2 r118, r92, r102;
}
{
sub.f16x2 r121, r115, r118;
}
{
mul.f16x2 r124, r89, r102;
}
{
fma.rn.f16x2 r127, r92, r101, r124;
}
{
neg.f16x2 r131, r86;
}
{
mul.f16x2 r133, r95, r105;
}
{
mul.f16x2 r136, r98, r106;
}
{
sub.f16x2 r139, r133, r136;
}
{
mul.f16x2 r142, r95, r106;
}
{
fma.rn.f16x2 r145, r98, r105, r142;
}
{
add.f16x2 r149, r27, r77;
}
{
add.f16x2 r152, r30, r80;
}
{
sub.f16x2 r155, r27, r77;
}
{
sub.f16x2 r158, r30, r80;
}
{
add.f16x2 r161, r39, r121;
}
{
add.f16x2 r164, r42, r127;
}
{
sub.f16x2 r167, r39, r121;
}
{
sub.f16x2 r170, r42, r127;
}
{
add.f16x2 r173, r33, r131;
}
{
add.f16x2 r176, r36, r83;
}
{
sub.f16x2 r179, r33, r131;
}
{
sub.f16x2 r182, r36, r83;
}
{
add.f16x2 r185, r45, r139;
}
{
add.f16x2 r188, r48, r145;
}
{
sub.f16x2 r191, r45, r139;
}
{
sub.f16x2 r194, r48, r145;
}
{
add.f16x2 r197, %35, %51;
}
{
add.f16x2 r200, %36, %52;
}
{
sub.f16x2 r203, %35, %51;
}
{
sub.f16x2 r206, %36, %52;
}
{
add.f16x2 r209, %43, %59;
}
{
add.f16x2 r212, %44, %60;
}
{
sub.f16x2 r215, %43, %59;
}
{
sub.f16x2 r218, %44, %60;
}
{
neg.f16x2 r221, r218;
}
{
add.f16x2 r223, r197, r209;
}
{
add.f16x2 r226, r200, r212;
}
{
sub.f16x2 r229, r197, r209;
}
{
sub.f16x2 r232, r200, r212;
}
{
add.f16x2 r235, r203, r221;
}
{
add.f16x2 r238, r206, r215;
}
{
sub.f16x2 r241, r203, r221;
}
{
sub.f16x2 r244, r206, r215;
}
{
add.f16x2 r247, %39, %55;
}
{
add.f16x2 r250, %40, %56;
}
{
sub.f16x2 r253, %39, %55;
}
{
sub.f16x2 r256, %40, %56;
}
{
add.f16x2 r259, %47, %63;
}
{
add.f16x2 r262, %48, %64;
}
{
sub.f16x2 r265, %47, %63;
}
{
sub.f16x2 r268, %48, %64;
}
{
neg.f16x2 r271, r268;
}
{
add.f16x2 r273, r247, r259;
}
{
add.f16x2 r276, r250, r262;
}
{
sub.f16x2 r279, r247, r259;
}
{
sub.f16x2 r282, r250, r262;
}
{
add.f16x2 r285, r253, r271;
}
{
add.f16x2 r288, r256, r265;
}
{
sub.f16x2 r291, r253, r271;
}
{
sub.f16x2 r294, r256, r265;
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r297, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r298, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r301, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r302, {low, high};
}
{
mul.f16x2 r311, r285, r297;
}
{
mul.f16x2 r314, r288, r298;
}
{
sub.f16x2 r317, r311, r314;
}
{
mul.f16x2 r320, r285, r298;
}
{
fma.rn.f16x2 r323, r288, r297, r320;
}
{
neg.f16x2 r327, r282;
}
{
mul.f16x2 r329, r291, r301;
}
{
mul.f16x2 r332, r294, r302;
}
{
sub.f16x2 r335, r329, r332;
}
{
mul.f16x2 r338, r291, r302;
}
{
fma.rn.f16x2 r341, r294, r301, r338;
}
{
add.f16x2 r345, r223, r273;
}
{
add.f16x2 r348, r226, r276;
}
{
sub.f16x2 r351, r223, r273;
}
{
sub.f16x2 r354, r226, r276;
}
{
add.f16x2 r357, r235, r317;
}
{
add.f16x2 r360, r238, r323;
}
{
sub.f16x2 r363, r235, r317;
}
{
sub.f16x2 r366, r238, r323;
}
{
add.f16x2 r369, r229, r327;
}
{
add.f16x2 r372, r232, r279;
}
{
sub.f16x2 r375, r229, r327;
}
{
sub.f16x2 r378, r232, r279;
}
{
add.f16x2 r381, r241, r335;
}
{
add.f16x2 r384, r244, r341;
}
{
sub.f16x2 r387, r241, r335;
}
{
sub.f16x2 r390, r244, r341;
}
mov.f32 f76, 0f3F6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r393, {low, high};
}
mov.f32 f84, 0f3EC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r394, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r395, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r396, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r397, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r398, {low, high};
}
mov.f32 f74, 0fBEC3EF15;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f74;
cvt.rn.f16.f32 high, f74;
mov.b32 r401, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f76;
cvt.rn.f16.f32 high, f76;
mov.b32 r402, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f78;
cvt.rn.f16.f32 high, f78;
mov.b32 r403, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f80;
cvt.rn.f16.f32 high, f80;
mov.b32 r404, {low, high};
}
mov.f32 f82, 0fBF6C835E;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f82;
cvt.rn.f16.f32 high, f82;
mov.b32 r405, {low, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f84;
cvt.rn.f16.f32 high, f84;
mov.b32 r406, {low, high};
}
{
mul.f16x2 r423, r357, r393;
}
{
mul.f16x2 r426, r360, r394;
}
{
sub.f16x2 r429, r423, r426;
}
{
mul.f16x2 r432, r357, r394;
}
{
fma.rn.f16x2 r435, r360, r393, r432;
}
{
mul.f16x2 r439, r369, r395;
}
{
mul.f16x2 r442, r372, r396;
}
{
sub.f16x2 r445, r439, r442;
}
{
mul.f16x2 r448, r369, r396;
}
{
fma.rn.f16x2 r451, r372, r395, r448;
}
{
mul.f16x2 r455, r381, r397;
}
{
mul.f16x2 r458, r384, r398;
}
{
sub.f16x2 r461, r455, r458;
}
{
mul.f16x2 r464, r381, r398;
}
{
fma.rn.f16x2 r467, r384, r397, r464;
}
{
neg.f16x2 r471, r354;
}
{
mul.f16x2 r473, r363, r401;
}
{
mul.f16x2 r476, r366, r402;
}
{
sub.f16x2 r479, r473, r476;
}
{
mul.f16x2 r482, r363, r402;
}
{
fma.rn.f16x2 r485, r366, r401, r482;
}
{
mul.f16x2 r489, r375, r403;
}
{
mul.f16x2 r492, r378, r404;
}
{
sub.f16x2 r495, r489, r492;
}
{
mul.f16x2 r498, r375, r404;
}
{
fma.rn.f16x2 r501, r378, r403, r498;
}
{
mul.f16x2 r505, r387, r405;
}
{
mul.f16x2 r508, r390, r406;
}
{
sub.f16x2 r511, r505, r508;
}
{
mul.f16x2 r514, r387, r406;
}
{
fma.rn.f16x2 r517, r390, r405, r514;
}
{
add.f16x2 r521, r149, r345;
}
{
add.f16x2 r524, r152, r348;
}
{
sub.f16x2 r527, r149, r345;
}
{
sub.f16x2 r530, r152, r348;
}
{
add.f16x2 r533, r161, r429;
}
{
add.f16x2 r536, r164, r435;
}
{
sub.f16x2 r539, r161, r429;
}
{
sub.f16x2 r542, r164, r435;
}
{
add.f16x2 r545, r173, r445;
}
{
add.f16x2 r548, r176, r451;
}
{
sub.f16x2 r551, r173, r445;
}
{
sub.f16x2 r554, r176, r451;
}
{
add.f16x2 r557, r185, r461;
}
{
add.f16x2 r560, r188, r467;
}
{
sub.f16x2 r563, r185, r461;
}
{
sub.f16x2 r566, r188, r467;
}
{
add.f16x2 r569, r155, r471;
}
{
add.f16x2 r572, r158, r351;
}
{
sub.f16x2 r575, r155, r471;
}
{
sub.f16x2 r578, r158, r351;
}
{
add.f16x2 r581, r167, r479;
}
{
add.f16x2 r584, r170, r485;
}
{
sub.f16x2 r587, r167, r479;
}
{
sub.f16x2 r590, r170, r485;
}
{
add.f16x2 r593, r179, r495;
}
{
add.f16x2 r596, r182, r501;
}
{
sub.f16x2 r599, r179, r495;
}
{
sub.f16x2 r602, r182, r501;
}
{
add.f16x2 r605, r191, r511;
}
{
add.f16x2 r608, r194, r517;
}
{
sub.f16x2 r611, r191, r511;
}
{
sub.f16x2 r614, r194, r517;
}
and.b32 r1276, r1275, 1;
shl.b32 r1277, r1275, 6;
and.b32 r1278, r1277, -128;
add.s32 r1279, r1274, r1278;
cvt.rn.f32.u32 f151, r1276;
mul.f32 f152, f151, 0f3E490FDB;
cos.approx.f32 f117, f152;
sin.approx.f32 f153, f152;
neg.f32 f118, f153;
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f117;
cvt.rn.f16.f32 high, f118;
mov.b32 r617, {low, high};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r620, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r622, {high, high};
}
{
mul.f16x2 r624, r536, r622;
}
{
fma.rn.f16x2 r627, r533, r620, r624;
}
{
mul.f16x2 r631, r533, r622;
}
{
neg.f16x2 r634, r631;
}
{
fma.rn.f16x2 r636, r536, r620, r634;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r640, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r642, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r644, {low, high};
}
{
mul.f16x2 r645, r642, r644;
}
{
mul.f16x2 r648, r617, r640;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r651, {high, low};
}
{
fma.rn.f16x2 r653, r645, r651, r648;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r657, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r659, {high, high};
}
{
mul.f16x2 r661, r548, r659;
}
{
fma.rn.f16x2 r664, r545, r657, r661;
}
{
mul.f16x2 r668, r545, r659;
}
{
neg.f16x2 r671, r668;
}
{
fma.rn.f16x2 r673, r548, r657, r671;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r677, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r679, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r681, {low, high};
}
{
mul.f16x2 r682, r679, r681;
}
{
mul.f16x2 r685, r653, r677;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r653;
mov.b32 r688, {high, low};
}
{
fma.rn.f16x2 r690, r682, r688, r685;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r694, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r696, {high, high};
}
{
mul.f16x2 r698, r560, r696;
}
{
fma.rn.f16x2 r701, r557, r694, r698;
}
{
mul.f16x2 r705, r557, r696;
}
{
neg.f16x2 r708, r705;
}
{
fma.rn.f16x2 r710, r560, r694, r708;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r714, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r716, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r718, {low, high};
}
{
mul.f16x2 r719, r716, r718;
}
{
mul.f16x2 r722, r690, r714;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r690;
mov.b32 r725, {high, low};
}
{
fma.rn.f16x2 r727, r719, r725, r722;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r731, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r733, {high, high};
}
{
mul.f16x2 r735, r572, r733;
}
{
fma.rn.f16x2 r738, r569, r731, r735;
}
{
mul.f16x2 r742, r569, r733;
}
{
neg.f16x2 r745, r742;
}
{
fma.rn.f16x2 r747, r572, r731, r745;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r751, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r753, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r755, {low, high};
}
{
mul.f16x2 r756, r753, r755;
}
{
mul.f16x2 r759, r727, r751;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r727;
mov.b32 r762, {high, low};
}
{
fma.rn.f16x2 r764, r756, r762, r759;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r768, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r770, {high, high};
}
{
mul.f16x2 r772, r584, r770;
}
{
fma.rn.f16x2 r775, r581, r768, r772;
}
{
mul.f16x2 r779, r581, r770;
}
{
neg.f16x2 r782, r779;
}
{
fma.rn.f16x2 r784, r584, r768, r782;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r788, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r790, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r792, {low, high};
}
{
mul.f16x2 r793, r790, r792;
}
{
mul.f16x2 r796, r764, r788;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r764;
mov.b32 r799, {high, low};
}
{
fma.rn.f16x2 r801, r793, r799, r796;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r805, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r807, {high, high};
}
{
mul.f16x2 r809, r596, r807;
}
{
fma.rn.f16x2 r812, r593, r805, r809;
}
{
mul.f16x2 r816, r593, r807;
}
{
neg.f16x2 r819, r816;
}
{
fma.rn.f16x2 r821, r596, r805, r819;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r825, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r827, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r829, {low, high};
}
{
mul.f16x2 r830, r827, r829;
}
{
mul.f16x2 r833, r801, r825;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r801;
mov.b32 r836, {high, low};
}
{
fma.rn.f16x2 r838, r830, r836, r833;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r842, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r844, {high, high};
}
{
mul.f16x2 r846, r608, r844;
}
{
fma.rn.f16x2 r849, r605, r842, r846;
}
{
mul.f16x2 r853, r605, r844;
}
{
neg.f16x2 r856, r853;
}
{
fma.rn.f16x2 r858, r608, r842, r856;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r862, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r864, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r866, {low, high};
}
{
mul.f16x2 r867, r864, r866;
}
{
mul.f16x2 r870, r838, r862;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r838;
mov.b32 r873, {high, low};
}
{
fma.rn.f16x2 r875, r867, r873, r870;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r879, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r881, {high, high};
}
{
mul.f16x2 r883, r530, r881;
}
{
fma.rn.f16x2 r886, r527, r879, r883;
}
{
mul.f16x2 r890, r527, r881;
}
{
neg.f16x2 r893, r890;
}
{
fma.rn.f16x2 r895, r530, r879, r893;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r899, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r901, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r903, {low, high};
}
{
mul.f16x2 r904, r901, r903;
}
{
mul.f16x2 r907, r875, r899;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r875;
mov.b32 r910, {high, low};
}
{
fma.rn.f16x2 r912, r904, r910, r907;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r916, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r918, {high, high};
}
{
mul.f16x2 r920, r542, r918;
}
{
fma.rn.f16x2 r923, r539, r916, r920;
}
{
mul.f16x2 r927, r539, r918;
}
{
neg.f16x2 r930, r927;
}
{
fma.rn.f16x2 r932, r542, r916, r930;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r936, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r938, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r940, {low, high};
}
{
mul.f16x2 r941, r938, r940;
}
{
mul.f16x2 r944, r912, r936;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r912;
mov.b32 r947, {high, low};
}
{
fma.rn.f16x2 r949, r941, r947, r944;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r953, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r955, {high, high};
}
{
mul.f16x2 r957, r554, r955;
}
{
fma.rn.f16x2 r960, r551, r953, r957;
}
{
mul.f16x2 r964, r551, r955;
}
{
neg.f16x2 r967, r964;
}
{
fma.rn.f16x2 r969, r554, r953, r967;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r973, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r975, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r977, {low, high};
}
{
mul.f16x2 r978, r975, r977;
}
{
mul.f16x2 r981, r949, r973;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r949;
mov.b32 r984, {high, low};
}
{
fma.rn.f16x2 r986, r978, r984, r981;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r990, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r992, {high, high};
}
{
mul.f16x2 r994, r566, r992;
}
{
fma.rn.f16x2 r997, r563, r990, r994;
}
{
mul.f16x2 r1001, r563, r992;
}
{
neg.f16x2 r1004, r1001;
}
{
fma.rn.f16x2 r1006, r566, r990, r1004;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1010, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1012, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1014, {low, high};
}
{
mul.f16x2 r1015, r1012, r1014;
}
{
mul.f16x2 r1018, r986, r1010;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r986;
mov.b32 r1021, {high, low};
}
{
fma.rn.f16x2 r1023, r1015, r1021, r1018;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1027, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1029, {high, high};
}
{
mul.f16x2 r1031, r578, r1029;
}
{
fma.rn.f16x2 r1034, r575, r1027, r1031;
}
{
mul.f16x2 r1038, r575, r1029;
}
{
neg.f16x2 r1041, r1038;
}
{
fma.rn.f16x2 r1043, r578, r1027, r1041;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1047, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1049, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1051, {low, high};
}
{
mul.f16x2 r1052, r1049, r1051;
}
{
mul.f16x2 r1055, r1023, r1047;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1023;
mov.b32 r1058, {high, low};
}
{
fma.rn.f16x2 r1060, r1052, r1058, r1055;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1064, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1066, {high, high};
}
{
mul.f16x2 r1068, r590, r1066;
}
{
fma.rn.f16x2 r1071, r587, r1064, r1068;
}
{
mul.f16x2 r1075, r587, r1066;
}
{
neg.f16x2 r1078, r1075;
}
{
fma.rn.f16x2 r1080, r590, r1064, r1078;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1084, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1086, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1088, {low, high};
}
{
mul.f16x2 r1089, r1086, r1088;
}
{
mul.f16x2 r1092, r1060, r1084;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1060;
mov.b32 r1095, {high, low};
}
{
fma.rn.f16x2 r1097, r1089, r1095, r1092;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1101, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1103, {high, high};
}
{
mul.f16x2 r1105, r602, r1103;
}
{
fma.rn.f16x2 r1108, r599, r1101, r1105;
}
{
mul.f16x2 r1112, r599, r1103;
}
{
neg.f16x2 r1115, r1112;
}
{
fma.rn.f16x2 r1117, r602, r1101, r1115;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1121, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r617;
mov.b32 r1123, {high, high};
}
{
.reg .f16 low, high;
cvt.rn.f16.f32 low, f147;
cvt.rn.f16.f32 high, f148;
mov.b32 r1125, {low, high};
}
{
mul.f16x2 r1126, r1123, r1125;
}
{
mul.f16x2 r1129, r1097, r1121;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1097;
mov.b32 r1132, {high, low};
}
{
fma.rn.f16x2 r1134, r1126, r1132, r1129;
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1138, {low, low};
}
{
.reg .f16 low, high;
mov.b32 {low, high}, r1134;
mov.b32 r1140, {high, high};
}
{
mul.f16x2 r1142, r614, r1140;
}
{
fma.rn.f16x2 r1145, r611, r1138, r1142;
}
{
mul.f16x2 r1149, r611, r1140;
}
{
neg.f16x2 r1152, r1149;
}
{
fma.rn.f16x2 r1154, r614, r1138, r1152;
}
barrier.sync 0;
and.b32 r1280, r1277, 64;
add.s32 r1281, r1279, r1280;
st.shared.v4.f32 [r1281], {r521, r627, r664, r701};
st.shared.v4.f32 [r1281+16], {r738, r775, r812, r849};
st.shared.v4.f32 [r1281+32], {r886, r923, r960, r997};
st.shared.v4.f32 [r1281+48], {r1034, r1071, r1108, r1145};
barrier.sync 0;
mad.lo.s32 r1282, r1276, -60, r1281;
ld.shared.u32 r1176, [r1282];
ld.shared.u32 r1188, [r1282+8];
ld.shared.u32 r1200, [r1282+16];
ld.shared.u32 r1212, [r1282+24];
ld.shared.u32 r1224, [r1282+32];
ld.shared.u32 r1236, [r1282+40];
ld.shared.u32 r1248, [r1282+48];
ld.shared.u32 r1260, [r1282+56];
ld.shared.u32 r1177, [r1282+64];
ld.shared.u32 r1189, [r1282+72];
ld.shared.u32 r1201, [r1282+80];
ld.shared.u32 r1213, [r1282+88];
ld.shared.u32 r1225, [r1282+96];
ld.shared.u32 r1237, [r1282+104];
ld.shared.u32 r1249, [r1282+112];
ld.shared.u32 r1261, [r1282+120];
barrier.sync 0;
st.shared.v4.f32 [r1281], {r524, r636, r673, r710};
st.shared.v4.f32 [r1281+16], {r747, r784, r821, r858};
st.shared.v4.f32 [r1281+32], {r895, r932, r969, r1006};
st.shared.v4.f32 [r1281+48], {r1043, r1080, r1117, r1154};
barrier.sync 0;
ld.shared.u32 r1179, [r1282];
ld.shared.u32 r1191, [r1282+8];
ld.shared.u32 r1203, [r1282+16];
ld.shared.u32 r1215, [r1282+24];
ld.shared.u32 r1227, [r1282+32];
ld.shared.u32 r1239, [r1282+40];
ld.shared.u32 r1251, [r1282+48];
ld.shared.u32 r1263, [r1282+56];
ld.shared.u32 r1180, [r1282+64];
ld.shared.u32 r1192, [r1282+72];
ld.shared.u32 r1204, [r1282+80];
ld.shared.u32 r1216, [r1282+88];
ld.shared.u32 r1228, [r1282+96];
ld.shared.u32 r1240, [r1282+104];
ld.shared.u32 r1252, [r1282+112];
ld.shared.u32 r1264, [r1282+120];
{
add.f16x2 %0, r1176, r1177;
}
{
add.f16x2 %1, r1179, r1180;
}
{
sub.f16x2 %16, r1176, r1177;
}
{
sub.f16x2 %17, r1179, r1180;
}
{
add.f16x2 %2, r1188, r1189;
}
{
add.f16x2 %3, r1191, r1192;
}
{
sub.f16x2 %18, r1188, r1189;
}
{
sub.f16x2 %19, r1191, r1192;
}
{
add.f16x2 %4, r1200, r1201;
}
{
add.f16x2 %5, r1203, r1204;
}
{
sub.f16x2 %20, r1200, r1201;
}
{
sub.f16x2 %21, r1203, r1204;
}
{
add.f16x2 %6, r1212, r1213;
}
{
add.f16x2 %7, r1215, r1216;
}
{
sub.f16x2 %22, r1212, r1213;
}
{
sub.f16x2 %23, r1215, r1216;
}
{
add.f16x2 %8, r1224, r1225;
}
{
add.f16x2 %9, r1227, r1228;
}
{
sub.f16x2 %24, r1224, r1225;
}
{
sub.f16x2 %25, r1227, r1228;
}
{
add.f16x2 %10, r1236, r1237;
}
{
add.f16x2 %11, r1239, r1240;
}
{
sub.f16x2 %26, r1236, r1237;
}
{
sub.f16x2 %27, r1239, r1240;
}
{
add.f16x2 %12, r1248, r1249;
}
{
add.f16x2 %13, r1251, r1252;
}
{
sub.f16x2 %28, r1248, r1249;
}
{
sub.f16x2 %29, r1251, r1252;
}
{
add.f16x2 %14, r1260, r1261;
}
{
add.f16x2 %15, r1263, r1264;
}
{
sub.f16x2 %30, r1260, r1261;
}
{
sub.f16x2 %31, r1263, r1264;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)), "=r"(__HALF2_TO_UI(rmem[6].x)), "=r"(__HALF2_TO_UI(rmem[6].y)), "=r"(__HALF2_TO_UI(rmem[7].x)), "=r"(__HALF2_TO_UI(rmem[7].y)), "=r"(__HALF2_TO_UI(rmem[8].x)), "=r"(__HALF2_TO_UI(rmem[8].y)), "=r"(__HALF2_TO_UI(rmem[9].x)), "=r"(__HALF2_TO_UI(rmem[9].y)), "=r"(__HALF2_TO_UI(rmem[10].x)), "=r"(__HALF2_TO_UI(rmem[10].y)), "=r"(__HALF2_TO_UI(rmem[11].x)), "=r"(__HALF2_TO_UI(rmem[11].y)), "=r"(__HALF2_TO_UI(rmem[12].x)), "=r"(__HALF2_TO_UI(rmem[12].y)), "=r"(__HALF2_TO_UI(rmem[13].x)), "=r"(__HALF2_TO_UI(rmem[13].y)), "=r"(__HALF2_TO_UI(rmem[14].x)), "=r"(__HALF2_TO_UI(rmem[14].y)), "=r"(__HALF2_TO_UI(rmem[15].x)), "=r"(__HALF2_TO_UI(rmem[15].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[6].x)), "r"(__HALF2_TO_UI(rmem[6].y)), "r"(__HALF2_TO_UI(rmem[7].x)), "r"(__HALF2_TO_UI(rmem[7].y)), "r"(__HALF2_TO_UI(rmem[8].x)), "r"(__HALF2_TO_UI(rmem[8].y)), "r"(__HALF2_TO_UI(rmem[9].x)), "r"(__HALF2_TO_UI(rmem[9].y)), "r"(__HALF2_TO_UI(rmem[10].x)), "r"(__HALF2_TO_UI(rmem[10].y)), "r"(__HALF2_TO_UI(rmem[11].x)), "r"(__HALF2_TO_UI(rmem[11].y)), "r"(__HALF2_TO_UI(rmem[12].x)), "r"(__HALF2_TO_UI(rmem[12].y)), "r"(__HALF2_TO_UI(rmem[13].x)), "r"(__HALF2_TO_UI(rmem[13].y)), "r"(__HALF2_TO_UI(rmem[14].x)), "r"(__HALF2_TO_UI(rmem[14].y)), "r"(__HALF2_TO_UI(rmem[15].x)), "r"(__HALF2_TO_UI(rmem[15].y)));
};


#endif
