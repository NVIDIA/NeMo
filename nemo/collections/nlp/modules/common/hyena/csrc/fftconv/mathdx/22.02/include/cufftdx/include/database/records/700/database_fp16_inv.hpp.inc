//Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    64,     0,   948>
    >;
};

template<> struct block_fft_record<14, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    64,     0,   949>
    >;
};

template<> struct block_fft_record<15, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    64,     0,   950>
    >;
};

template<> struct block_fft_record<17, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    64,     0,   951>
    >;
};

template<> struct block_fft_record<18, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    64,     0,   952>
    >;
};

template<> struct block_fft_record<19, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    64,     0,   953>
    >;
};

template<> struct block_fft_record<20, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    64,     0,   954>
    >;
};

template<> struct block_fft_record<21, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    64,     0,   955>
    >;
};

template<> struct block_fft_record<22, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    64,     0,   956>
    >;
};

template<> struct block_fft_record<23, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    64,     0,   957>
    >;
};

template<> struct block_fft_record<24, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    64,     0,   958>
    >;
};

template<> struct block_fft_record<26, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    64,     0,   959>
    >;
};

template<> struct block_fft_record<28, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    64,     0,   960>
    >;
};

template<> struct block_fft_record<29, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    64,     0,   961>
    >;
};

template<> struct block_fft_record<30, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    64,     0,   962>
    >;
};

template<> struct block_fft_record<31, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    64,     0,   963>
    >;
};

template<> struct block_fft_record<2, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,   128,     0,   964>
    >;
};

template<> struct block_fft_record<4, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,   128,     0,   965>,
        block_fft_implementation< 2,    2,    2,   256,     8,   966>,
        block_fft_implementation< 2,    2,    2,   256,    16,   967>
    >;
};

template<> struct block_fft_record<8, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    64,     0,   968>,
        block_fft_implementation< 4,    4,    2,   256,    16,   969>,
        block_fft_implementation< 4,    4,    2,   192,    32,   970>,
        block_fft_implementation< 2,    2,    4,   160,    16,   971>,
        block_fft_implementation< 2,    2,    4,   256,    32,   972>
    >;
};

template<> struct block_fft_record<16, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,   224,     0,   973>,
        block_fft_implementation< 4,    4,    4,   128,    32,   974>,
        block_fft_implementation< 4,    4,    4,   192,    64,   975>,
        block_fft_implementation< 8,    8,    2,   256,    32,   976>,
        block_fft_implementation< 8,    8,    2,   224,    64,   977>,
        block_fft_implementation< 2,    2,    8,   128,    32,   978>,
        block_fft_implementation< 2,    2,    8,   128,    64,   979>
    >;
};

template<> struct block_fft_record<32, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,    1,    64,     0,   980>,
        block_fft_implementation< 8,    8,    4,    96,   128,   981>,
        block_fft_implementation< 8,    8,    4,   192,    64,   982>,
        block_fft_implementation< 4,    4,    8,    96,   128,   983>,
        block_fft_implementation< 4,    4,    8,   128,    64,   984>,
        block_fft_implementation< 2,    2,   16,    40,   128,   985>,
        block_fft_implementation<16,   16,    2,   128,   128,   986>,
        block_fft_implementation< 2,    2,   16,    52,    64,   987>,
        block_fft_implementation<16,   16,    2,   256,    64,   988>
    >;
};

template<> struct block_fft_record<64, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    48,   256,   989>,
        block_fft_implementation< 8,    8,    8,    26,   128,   990>,
        block_fft_implementation< 4,    4,   16,    18,   256,   991>,
        block_fft_implementation< 4,    4,   16,    30,   128,   992>,
        block_fft_implementation<16,   16,    4,    50,   256,   993>,
        block_fft_implementation<16,   16,    4,    58,   128,   994>,
        block_fft_implementation<32,   32,    2,    46,   256,   995>,
        block_fft_implementation< 2,    2,   32,    28,   256,   996>,
        block_fft_implementation< 2,    2,   32,    16,   128,   997>,
        block_fft_implementation<32,   32,    2,   128,   128,   998>
    >;
};

template<> struct block_fft_record<128, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    8,    32,   256,   999>,
        block_fft_implementation< 8,    8,   16,    16,   512,  1000>,
        block_fft_implementation<16,   16,    8,    26,   512,  1001>,
        block_fft_implementation< 8,    8,   16,    32,   256,  1002>,
        block_fft_implementation< 4,    4,   32,     8,   512,  1003>,
        block_fft_implementation< 4,    4,   32,    14,   256,  1004>,
        block_fft_implementation<32,   32,    4,     8,   512,  1005>,
        block_fft_implementation<32,   32,    4,    62,   256,  1006>,
        block_fft_implementation< 2,    2,   64,    12,   512,  1007>,
        block_fft_implementation< 2,    2,   64,    12,   256,  1008>
    >;
};

template<> struct block_fft_record<256, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,     4,   512,  1009>,
        block_fft_implementation< 8,    8,   32,     8,  1024,  1010>,
        block_fft_implementation< 8,    8,   32,     8,   512,  1011>,
        block_fft_implementation<16,   16,   16,     6,  1024,  1012>,
        block_fft_implementation<32,   32,    8,     8,  1024,  1013>,
        block_fft_implementation< 4,    4,   64,     4,  1024,  1014>,
        block_fft_implementation< 4,    4,   64,     8,   512,  1015>,
        block_fft_implementation<32,   32,    8,    32,   512,  1016>,
        block_fft_implementation< 2,    2,  128,     6,  1024,  1017>,
        block_fft_implementation< 2,    2,  128,     8,   512,  1018>
    >;
};

template<> struct block_fft_record<512, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     4,  2048,  1019>,
        block_fft_implementation< 8,    8,   64,     4,  1024,  1020>,
        block_fft_implementation<16,   16,   32,     2,  1024,  1021>,
        block_fft_implementation<32,   32,   16,     4,  2048,  1022>,
        block_fft_implementation<32,   32,   16,     4,  1024,  1023>,
        block_fft_implementation<16,   16,   32,     8,  2048,  1024>,
        block_fft_implementation< 4,    4,  128,     4,  2048,  1025>,
        block_fft_implementation< 4,    4,  128,     4,  1024,  1026>,
        block_fft_implementation< 2,    2,  256,     2,  2048,  1027>,
        block_fft_implementation< 2,    2,  256,     4,  1024,  1028>
    >;
};

template<> struct block_fft_record<1024, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   64,     2,  2048,  1029>,
        block_fft_implementation< 8,    8,  128,     2,  4096,  1030>,
        block_fft_implementation<32,   32,   32,     8,  4096,  1031>,
        block_fft_implementation< 8,    8,  128,     2,  2048,  1032>,
        block_fft_implementation<32,   32,   32,     8,  2048,  1033>,
        block_fft_implementation<16,   16,   64,     4,  4096,  1034>,
        block_fft_implementation< 4,    4,  256,     2,  4096,  1035>,
        block_fft_implementation< 4,    4,  256,     2,  2048,  1036>,
        block_fft_implementation< 2,    2,  512,     2,  4096,  1037>,
        block_fft_implementation< 2,    2,  512,     2,  2048,  1038>
    >;
};

template<> struct block_fft_record<2048, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  128,     2,  4096,  1039>,
        block_fft_implementation< 8,    8,  256,     2,  8192,  1040>,
        block_fft_implementation< 8,    8,  256,     2,  4096,  1041>,
        block_fft_implementation<16,   16,  128,     2,  8192,  1042>,
        block_fft_implementation<32,   32,   64,     4,  8192,  1043>,
        block_fft_implementation<32,   32,   64,     4,  4096,  1044>,
        block_fft_implementation< 4,    4,  512,     2,  8192,  1045>,
        block_fft_implementation< 4,    4,  512,     2,  4096,  1046>,
        block_fft_implementation< 2,    2, 1024,     2,  8192,  1047>,
        block_fft_implementation< 2,    2, 1024,     2,  4096,  1048>
    >;
};

template<> struct block_fft_record<4096, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  256,     2,  8192,  1049>,
        block_fft_implementation< 8,    8,  512,     2, 16384,  1050>,
        block_fft_implementation< 8,    8,  512,     2,  8192,  1051>,
        block_fft_implementation<16,   16,  256,     2, 16384,  1052>,
        block_fft_implementation<32,   32,  128,     2, 16384,  1053>,
        block_fft_implementation<32,   32,  128,     2,  8192,  1054>,
        block_fft_implementation< 4,    4, 1024,     2, 16384,  1055>,
        block_fft_implementation< 4,    4, 1024,     2,  8192,  1056>
    >;
};

template<> struct block_fft_record<8192, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  512,     2, 16384,  1057>,
        block_fft_implementation<32,   32,  256,     2, 32768,  1058>,
        block_fft_implementation<32,   32,  256,     2, 16384,  1059>,
        block_fft_implementation< 8,    8, 1024,     2, 32768,  1060>,
        block_fft_implementation<16,   16,  512,     2, 32768,  1061>,
        block_fft_implementation< 8,    8, 1024,     2, 16384,  1062>
    >;
};

template<> struct block_fft_record<16384, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16, 1024,     2, 32768,  1063>,
        block_fft_implementation<32,   32,  512,     2, 32768,  1064>
    >;
};

template<> struct block_fft_record<3, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,   128,     0,  1065>
    >;
};

template<> struct block_fft_record<9, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,   128,     0,  1066>,
        block_fft_implementation< 3,    3,    3,   128,    36,  1067>,
        block_fft_implementation< 3,    3,    3,   256,    18,  1068>
    >;
};

template<> struct block_fft_record<27, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,    64,     0,  1069>,
        block_fft_implementation< 9,    9,    3,    42,    54,  1070>,
        block_fft_implementation< 9,    9,    3,    20,   108,  1071>,
        block_fft_implementation< 3,    3,    9,    52,   108,  1072>,
        block_fft_implementation< 3,    3,    9,    48,    54,  1073>
    >;
};

template<> struct block_fft_record<81, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    28,   324,  1074>,
        block_fft_implementation< 9,    9,    9,    14,   162,  1075>,
        block_fft_implementation<27,   27,    3,    20,   162,  1076>,
        block_fft_implementation<27,   27,    3,    18,   324,  1077>,
        block_fft_implementation< 3,    3,   27,    14,   324,  1078>,
        block_fft_implementation< 3,    3,   27,    18,   162,  1079>
    >;
};

template<> struct block_fft_record<243, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    9,    28,   972,  1080>,
        block_fft_implementation<27,   27,    9,    28,   486,  1081>,
        block_fft_implementation< 9,    9,   27,    18,   972,  1082>,
        block_fft_implementation< 9,    9,   27,    18,   486,  1083>,
        block_fft_implementation< 3,    3,   81,     6,   972,  1084>,
        block_fft_implementation< 3,    3,   81,     6,   486,  1085>
    >;
};

template<> struct block_fft_record<729, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    28,  2916,  1086>,
        block_fft_implementation<27,   27,   27,    18,  1458,  1087>,
        block_fft_implementation< 9,    9,   81,     6,  2916,  1088>,
        block_fft_implementation< 9,    9,   81,     6,  1458,  1089>,
        block_fft_implementation< 3,    3,  243,     2,  2916,  1090>,
        block_fft_implementation< 3,    3,  243,     2,  1458,  1091>
    >;
};

template<> struct block_fft_record<2187, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  243,     2,  8748,  1092>,
        block_fft_implementation<27,   27,   81,     6,  8748,  1093>,
        block_fft_implementation< 9,    9,  243,     2,  4374,  1094>,
        block_fft_implementation<27,   27,   81,     6,  4374,  1095>,
        block_fft_implementation< 3,    3,  729,     2,  8748,  1096>,
        block_fft_implementation< 3,    3,  729,     2,  4374,  1097>
    >;
};

template<> struct block_fft_record<6561, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,  243,     2, 13122,  1098>,
        block_fft_implementation<27,   27,  243,     2, 26244,  1099>,
        block_fft_implementation< 9,    9,  729,     2, 13122,  1100>,
        block_fft_implementation< 9,    9,  729,     2, 26244,  1101>
    >;
};

template<> struct block_fft_record<5, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    64,     0,  1102>
    >;
};

template<> struct block_fft_record<25, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,    64,     0,  1103>,
        block_fft_implementation< 5,    5,    5,   256,   100,  1104>,
        block_fft_implementation< 5,    5,    5,    50,    50,  1105>
    >;
};

template<> struct block_fft_record<125, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    5,    50,   250,  1106>,
        block_fft_implementation<25,   25,    5,    50,   500,  1107>,
        block_fft_implementation< 5,    5,   25,    10,   500,  1108>,
        block_fft_implementation< 5,    5,   25,    10,   250,  1109>
    >;
};

template<> struct block_fft_record<625, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,    10,  1250,  1110>,
        block_fft_implementation<25,   25,   25,    10,  2500,  1111>,
        block_fft_implementation< 5,    5,  125,     2,  2500,  1112>,
        block_fft_implementation< 5,    5,  125,     2,  1250,  1113>
    >;
};

template<> struct block_fft_record<3125, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  125,     2,  6250,  1114>,
        block_fft_implementation<25,   25,  125,     6, 12500,  1115>,
        block_fft_implementation< 5,    5,  625,     2, 12500,  1116>,
        block_fft_implementation< 5,    5,  625,     2,  6250,  1117>
    >;
};

template<> struct block_fft_record<15625, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  625,     2, 31250,  1118>
    >;
};

template<> struct block_fft_record<7, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,   128,     0,  1119>
    >;
};

template<> struct block_fft_record<49, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    18,   196,  1120>,
        block_fft_implementation< 7,    7,    7,    18,    98,  1121>
    >;
};

template<> struct block_fft_record<343, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,    10,  1372,  1122>,
        block_fft_implementation< 7,    7,   49,    10,   686,  1123>
    >;
};

template<> struct block_fft_record<2401, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     2,  9604,  1124>,
        block_fft_implementation< 7,    7,  343,     2,  4802,  1125>
    >;
};

template<> struct block_fft_record<11, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,   192,     0,  1126>
    >;
};

template<> struct block_fft_record<121, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    46,   484,  1127>,
        block_fft_implementation<11,   11,   11,    46,   242,  1128>
    >;
};

template<> struct block_fft_record<1331, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     2,  2662,  1129>,
        block_fft_implementation<11,   11,  121,     2,  5324,  1130>
    >;
};

template<> struct block_fft_record<6, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    64,     0,  1131>
    >;
};

template<> struct block_fft_record<36, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,   128,    72,  1132>,
        block_fft_implementation< 6,    6,    6,   128,   144,  1133>
    >;
};

template<> struct block_fft_record<216, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,    14,   864,  1134>,
        block_fft_implementation< 6,    6,   36,    14,   432,  1135>
    >;
};

template<> struct block_fft_record<1296, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     4,  5184,  1136>,
        block_fft_implementation< 6,    6,  216,     2,  2592,  1137>
    >;
};

template<> struct block_fft_record<10, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    64,     0,  1138>
    >;
};

template<> struct block_fft_record<100, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    50,   200,  1139>,
        block_fft_implementation<10,   10,   10,    50,   400,  1140>
    >;
};

template<> struct block_fft_record<1000, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,    10,  4000,  1141>,
        block_fft_implementation<10,   10,  100,    10,  2000,  1142>
    >;
};

template<> struct block_fft_record<10000, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10, 1000,     2, 40000,  1143>,
        block_fft_implementation<10,   10, 1000,     2, 20000,  1144>
    >;
};

template<> struct block_fft_record<12, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,   128,     0,  1145>
    >;
};

template<> struct block_fft_record<144, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    42,   576,  1146>,
        block_fft_implementation<12,   12,   12,    42,   288,  1147>
    >;
};

template<> struct block_fft_record<1728, half, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     2,  6912,  1148>,
        block_fft_implementation<12,   12,  144,     2,  3456,  1149>
    >;
};
