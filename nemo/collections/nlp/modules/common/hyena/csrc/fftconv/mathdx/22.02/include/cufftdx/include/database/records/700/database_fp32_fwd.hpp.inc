//Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    32,     0,     0>
    >;
};

template<> struct block_fft_record<14, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    32,     0,     1>
    >;
};

template<> struct block_fft_record<15, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    32,     0,     2>
    >;
};

template<> struct block_fft_record<17, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    32,     0,     3>
    >;
};

template<> struct block_fft_record<18, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    32,     0,     4>
    >;
};

template<> struct block_fft_record<19, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    32,     0,     5>
    >;
};

template<> struct block_fft_record<20, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    32,     0,     6>
    >;
};

template<> struct block_fft_record<21, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    32,     0,     7>
    >;
};

template<> struct block_fft_record<22, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    32,     0,     8>
    >;
};

template<> struct block_fft_record<23, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    32,     0,     9>
    >;
};

template<> struct block_fft_record<24, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    32,     0,    10>
    >;
};

template<> struct block_fft_record<26, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    32,     0,    11>
    >;
};

template<> struct block_fft_record<28, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    32,     0,    12>
    >;
};

template<> struct block_fft_record<29, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    32,     0,    13>
    >;
};

template<> struct block_fft_record<30, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    32,     0,    14>
    >;
};

template<> struct block_fft_record<31, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    32,     0,    15>
    >;
};

template<> struct block_fft_record<2, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,    64,     0,    16>
    >;
};

template<> struct block_fft_record<4, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,    64,     0,    17>,
        block_fft_implementation< 2,    2,    2,   128,    16,    18>,
        block_fft_implementation< 2,    2,    2,   128,    32,    19>
    >;
};

template<> struct block_fft_record<8, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    32,     0,    20>,
        block_fft_implementation< 4,    4,    2,   128,    32,    21>,
        block_fft_implementation< 4,    4,    2,    96,    64,    22>,
        block_fft_implementation< 2,    2,    4,    80,    32,    23>,
        block_fft_implementation< 2,    2,    4,   128,    64,    24>
    >;
};

template<> struct block_fft_record<16, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,   112,     0,    25>,
        block_fft_implementation< 4,    4,    4,    64,    64,    26>,
        block_fft_implementation< 4,    4,    4,    96,   128,    27>,
        block_fft_implementation< 8,    8,    2,   128,    64,    28>,
        block_fft_implementation< 8,    8,    2,   112,   128,    29>,
        block_fft_implementation< 2,    2,    8,    64,    64,    30>,
        block_fft_implementation< 2,    2,    8,    64,   128,    31>
    >;
};

template<> struct block_fft_record<32, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,    1,    32,     0,    32>,
        block_fft_implementation< 8,    8,    4,    48,   256,    33>,
        block_fft_implementation< 8,    8,    4,    96,   128,    34>,
        block_fft_implementation< 4,    4,    8,    48,   256,    35>,
        block_fft_implementation< 4,    4,    8,    64,   128,    36>,
        block_fft_implementation< 2,    2,   16,    20,   256,    37>,
        block_fft_implementation<16,   16,    2,    64,   256,    38>,
        block_fft_implementation< 2,    2,   16,    26,   128,    39>,
        block_fft_implementation<16,   16,    2,   128,   128,    40>
    >;
};

template<> struct block_fft_record<64, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    24,   512,    41>,
        block_fft_implementation< 8,    8,    8,    13,   256,    42>,
        block_fft_implementation< 4,    4,   16,     9,   512,    43>,
        block_fft_implementation< 4,    4,   16,    15,   256,    44>,
        block_fft_implementation<16,   16,    4,    25,   512,    45>,
        block_fft_implementation<16,   16,    4,    29,   256,    46>,
        block_fft_implementation<32,   32,    2,    23,   512,    47>,
        block_fft_implementation< 2,    2,   32,    14,   512,    48>,
        block_fft_implementation< 2,    2,   32,     8,   256,    49>,
        block_fft_implementation<32,   32,    2,    64,   256,    50>
    >;
};

template<> struct block_fft_record<128, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    8,    16,   512,    51>,
        block_fft_implementation< 8,    8,   16,     8,  1024,    52>,
        block_fft_implementation<16,   16,    8,    13,  1024,    53>,
        block_fft_implementation< 8,    8,   16,    16,   512,    54>,
        block_fft_implementation< 4,    4,   32,     4,  1024,    55>,
        block_fft_implementation< 4,    4,   32,     7,   512,    56>,
        block_fft_implementation<32,   32,    4,     4,  1024,    57>,
        block_fft_implementation<32,   32,    4,    31,   512,    58>,
        block_fft_implementation< 2,    2,   64,     6,  1024,    59>,
        block_fft_implementation< 2,    2,   64,     6,   512,    60>
    >;
};

template<> struct block_fft_record<256, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,     2,  1024,    61>,
        block_fft_implementation< 8,    8,   32,     4,  2048,    62>,
        block_fft_implementation< 8,    8,   32,     4,  1024,    63>,
        block_fft_implementation<16,   16,   16,     3,  2048,    64>,
        block_fft_implementation<32,   32,    8,     4,  2048,    65>,
        block_fft_implementation< 4,    4,   64,     2,  2048,    66>,
        block_fft_implementation< 4,    4,   64,     4,  1024,    67>,
        block_fft_implementation<32,   32,    8,    16,  1024,    68>,
        block_fft_implementation< 2,    2,  128,     3,  2048,    69>,
        block_fft_implementation< 2,    2,  128,     4,  1024,    70>
    >;
};

template<> struct block_fft_record<512, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     2,  4096,    71>,
        block_fft_implementation< 8,    8,   64,     2,  2048,    72>,
        block_fft_implementation<16,   16,   32,     1,  2048,    73>,
        block_fft_implementation<32,   32,   16,     2,  4096,    74>,
        block_fft_implementation<32,   32,   16,     2,  2048,    75>,
        block_fft_implementation<16,   16,   32,     4,  4096,    76>,
        block_fft_implementation< 4,    4,  128,     2,  4096,    77>,
        block_fft_implementation< 4,    4,  128,     2,  2048,    78>,
        block_fft_implementation< 2,    2,  256,     1,  4096,    79>,
        block_fft_implementation< 2,    2,  256,     2,  2048,    80>
    >;
};

template<> struct block_fft_record<1024, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   64,     1,  4096,    81>,
        block_fft_implementation< 8,    8,  128,     1,  8192,    82>,
        block_fft_implementation<32,   32,   32,     4,  8192,    83>,
        block_fft_implementation< 8,    8,  128,     1,  4096,    84>,
        block_fft_implementation<32,   32,   32,     4,  4096,    85>,
        block_fft_implementation<16,   16,   64,     2,  8192,    86>,
        block_fft_implementation< 4,    4,  256,     1,  8192,    87>,
        block_fft_implementation< 4,    4,  256,     1,  4096,    88>,
        block_fft_implementation< 2,    2,  512,     1,  8192,    89>,
        block_fft_implementation< 2,    2,  512,     1,  4096,    90>
    >;
};

template<> struct block_fft_record<2048, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  128,     1,  8192,    91>,
        block_fft_implementation< 8,    8,  256,     1, 16384,    92>,
        block_fft_implementation< 8,    8,  256,     1,  8192,    93>,
        block_fft_implementation<16,   16,  128,     1, 16384,    94>,
        block_fft_implementation<32,   32,   64,     2, 16384,    95>,
        block_fft_implementation<32,   32,   64,     2,  8192,    96>,
        block_fft_implementation< 4,    4,  512,     1, 16384,    97>,
        block_fft_implementation< 4,    4,  512,     1,  8192,    98>,
        block_fft_implementation< 2,    2, 1024,     1, 16384,    99>,
        block_fft_implementation< 2,    2, 1024,     1,  8192,   100>
    >;
};

template<> struct block_fft_record<4096, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  256,     1, 16384,   101>,
        block_fft_implementation< 8,    8,  512,     1, 32768,   102>,
        block_fft_implementation< 8,    8,  512,     1, 16384,   103>,
        block_fft_implementation<16,   16,  256,     1, 32768,   104>,
        block_fft_implementation<32,   32,  128,     1, 32768,   105>,
        block_fft_implementation<32,   32,  128,     1, 16384,   106>,
        block_fft_implementation< 4,    4, 1024,     1, 32768,   107>,
        block_fft_implementation< 4,    4, 1024,     1, 16384,   108>
    >;
};

template<> struct block_fft_record<8192, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  512,     1, 32768,   109>,
        block_fft_implementation<32,   32,  256,     1, 65536,   110>,
        block_fft_implementation<32,   32,  256,     1, 32768,   111>,
        block_fft_implementation< 8,    8, 1024,     1, 65536,   112>,
        block_fft_implementation<16,   16,  512,     1, 65536,   113>,
        block_fft_implementation< 8,    8, 1024,     1, 32768,   114>
    >;
};

template<> struct block_fft_record<16384, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16, 1024,     1, 65536,   115>,
        block_fft_implementation<32,   32,  512,     1, 65536,   116>
    >;
};

template<> struct block_fft_record<3, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,    64,     0,   117>
    >;
};

template<> struct block_fft_record<9, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,    64,     0,   118>,
        block_fft_implementation< 3,    3,    3,    64,    72,   119>,
        block_fft_implementation< 3,    3,    3,   128,    36,   120>
    >;
};

template<> struct block_fft_record<27, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,    32,     0,   121>,
        block_fft_implementation< 9,    9,    3,    21,   108,   122>,
        block_fft_implementation< 9,    9,    3,    10,   216,   123>,
        block_fft_implementation< 3,    3,    9,    26,   216,   124>,
        block_fft_implementation< 3,    3,    9,    24,   108,   125>
    >;
};

template<> struct block_fft_record<81, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    14,   648,   126>,
        block_fft_implementation< 9,    9,    9,     7,   324,   127>,
        block_fft_implementation<27,   27,    3,    10,   324,   128>,
        block_fft_implementation<27,   27,    3,     9,   648,   129>,
        block_fft_implementation< 3,    3,   27,     7,   648,   130>,
        block_fft_implementation< 3,    3,   27,     9,   324,   131>
    >;
};

template<> struct block_fft_record<243, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    9,    14,  1944,   132>,
        block_fft_implementation<27,   27,    9,    14,   972,   133>,
        block_fft_implementation< 9,    9,   27,     9,  1944,   134>,
        block_fft_implementation< 9,    9,   27,     9,   972,   135>,
        block_fft_implementation< 3,    3,   81,     3,  1944,   136>,
        block_fft_implementation< 3,    3,   81,     3,   972,   137>
    >;
};

template<> struct block_fft_record<729, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    14,  5832,   138>,
        block_fft_implementation<27,   27,   27,     9,  2916,   139>,
        block_fft_implementation< 9,    9,   81,     3,  5832,   140>,
        block_fft_implementation< 9,    9,   81,     3,  2916,   141>,
        block_fft_implementation< 3,    3,  243,     1,  5832,   142>,
        block_fft_implementation< 3,    3,  243,     1,  2916,   143>
    >;
};

template<> struct block_fft_record<2187, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  243,     1, 17496,   144>,
        block_fft_implementation<27,   27,   81,     3, 17496,   145>,
        block_fft_implementation< 9,    9,  243,     1,  8748,   146>,
        block_fft_implementation<27,   27,   81,     3,  8748,   147>,
        block_fft_implementation< 3,    3,  729,     1, 17496,   148>,
        block_fft_implementation< 3,    3,  729,     1,  8748,   149>
    >;
};

template<> struct block_fft_record<6561, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,  243,     1, 26244,   150>,
        block_fft_implementation<27,   27,  243,     1, 52488,   151>,
        block_fft_implementation< 9,    9,  729,     1, 26244,   152>,
        block_fft_implementation< 9,    9,  729,     1, 52488,   153>
    >;
};

template<> struct block_fft_record<5, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    32,     0,   154>
    >;
};

template<> struct block_fft_record<25, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,    32,     0,   155>,
        block_fft_implementation< 5,    5,    5,   128,   200,   156>,
        block_fft_implementation< 5,    5,    5,    25,   100,   157>
    >;
};

template<> struct block_fft_record<125, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    5,    25,   500,   158>,
        block_fft_implementation<25,   25,    5,    25,  1000,   159>,
        block_fft_implementation< 5,    5,   25,     5,  1000,   160>,
        block_fft_implementation< 5,    5,   25,     5,   500,   161>
    >;
};

template<> struct block_fft_record<625, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,     5,  2500,   162>,
        block_fft_implementation<25,   25,   25,     5,  5000,   163>,
        block_fft_implementation< 5,    5,  125,     1,  5000,   164>,
        block_fft_implementation< 5,    5,  125,     1,  2500,   165>
    >;
};

template<> struct block_fft_record<3125, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  125,     1, 12500,   166>,
        block_fft_implementation<25,   25,  125,     3, 25000,   167>,
        block_fft_implementation< 5,    5,  625,     1, 25000,   168>,
        block_fft_implementation< 5,    5,  625,     1, 12500,   169>
    >;
};

template<> struct block_fft_record<15625, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  625,     1, 62500,   170>
    >;
};

template<> struct block_fft_record<7, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,    64,     0,   171>
    >;
};

template<> struct block_fft_record<49, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,     9,   392,   172>,
        block_fft_implementation< 7,    7,    7,     9,   196,   173>
    >;
};

template<> struct block_fft_record<343, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,     5,  2744,   174>,
        block_fft_implementation< 7,    7,   49,     5,  1372,   175>
    >;
};

template<> struct block_fft_record<2401, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     1, 19208,   176>,
        block_fft_implementation< 7,    7,  343,     1,  9604,   177>
    >;
};

template<> struct block_fft_record<11, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,    96,     0,   178>
    >;
};

template<> struct block_fft_record<121, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    23,   968,   179>,
        block_fft_implementation<11,   11,   11,    23,   484,   180>
    >;
};

template<> struct block_fft_record<1331, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     1,  5324,   181>,
        block_fft_implementation<11,   11,  121,     1, 10648,   182>
    >;
};

template<> struct block_fft_record<6, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    32,     0,   183>
    >;
};

template<> struct block_fft_record<36, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    64,   144,   184>,
        block_fft_implementation< 6,    6,    6,    64,   288,   185>
    >;
};

template<> struct block_fft_record<216, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,     7,  1728,   186>,
        block_fft_implementation< 6,    6,   36,     7,   864,   187>
    >;
};

template<> struct block_fft_record<1296, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     2, 10368,   188>,
        block_fft_implementation< 6,    6,  216,     1,  5184,   189>
    >;
};

template<> struct block_fft_record<10, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    32,     0,   190>
    >;
};

template<> struct block_fft_record<100, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    25,   400,   191>,
        block_fft_implementation<10,   10,   10,    25,   800,   192>
    >;
};

template<> struct block_fft_record<1000, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,     5,  8000,   193>,
        block_fft_implementation<10,   10,  100,     5,  4000,   194>
    >;
};

template<> struct block_fft_record<10000, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10, 1000,     1, 80000,   195>,
        block_fft_implementation<10,   10, 1000,     1, 40000,   196>
    >;
};

template<> struct block_fft_record<12, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,    64,     0,   197>
    >;
};

template<> struct block_fft_record<144, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    21,  1152,   198>,
        block_fft_implementation<12,   12,   12,    21,   576,   199>
    >;
};

template<> struct block_fft_record<1728, float, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     1, 13824,   200>,
        block_fft_implementation<12,   12,  144,     1,  6912,   201>
    >;
};
