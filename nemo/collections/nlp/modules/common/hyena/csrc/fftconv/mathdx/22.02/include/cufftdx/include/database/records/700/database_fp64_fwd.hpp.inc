//Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    32,     0,   404>
    >;
};

template<> struct block_fft_record<14, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    32,     0,   405>
    >;
};

template<> struct block_fft_record<15, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    32,     0,   406>
    >;
};

template<> struct block_fft_record<17, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    32,     0,   407>
    >;
};

template<> struct block_fft_record<18, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    32,     0,   408>
    >;
};

template<> struct block_fft_record<19, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    32,     0,   409>
    >;
};

template<> struct block_fft_record<20, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    32,     0,   410>
    >;
};

template<> struct block_fft_record<21, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    32,     0,   411>
    >;
};

template<> struct block_fft_record<22, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    32,     0,   412>
    >;
};

template<> struct block_fft_record<23, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    32,     0,   413>
    >;
};

template<> struct block_fft_record<24, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    32,     0,   414>
    >;
};

template<> struct block_fft_record<26, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    32,     0,   415>
    >;
};

template<> struct block_fft_record<28, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    32,     0,   416>
    >;
};

template<> struct block_fft_record<29, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    32,     0,   417>
    >;
};

template<> struct block_fft_record<30, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    32,     0,   418>
    >;
};

template<> struct block_fft_record<31, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    32,     0,   419>
    >;
};

template<> struct block_fft_record<2, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,    32,     0,   420>
    >;
};

template<> struct block_fft_record<4, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,    96,     0,   421>,
        block_fft_implementation< 2,    2,    2,   128,    32,   422>,
        block_fft_implementation< 2,    2,    2,   128,    64,   423>
    >;
};

template<> struct block_fft_record<8, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    32,     0,   424>,
        block_fft_implementation< 4,    4,    2,   128,    64,   425>,
        block_fft_implementation< 2,    2,    4,   128,    64,   426>,
        block_fft_implementation< 4,    4,    2,   128,   128,   427>,
        block_fft_implementation< 2,    2,    4,   112,   128,   428>
    >;
};

template<> struct block_fft_record<16, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,    32,     0,   429>,
        block_fft_implementation< 4,    4,    4,    64,   256,   430>,
        block_fft_implementation< 4,    4,    4,    64,   128,   431>,
        block_fft_implementation< 2,    2,    8,    48,   256,   432>,
        block_fft_implementation< 8,    8,    2,   128,   128,   433>,
        block_fft_implementation< 8,    8,    2,    64,   256,   434>,
        block_fft_implementation< 2,    2,    8,    64,   128,   435>
    >;
};

template<> struct block_fft_record<32, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    4,    64,   256,   436>,
        block_fft_implementation< 4,    4,    8,    24,   512,   437>,
        block_fft_implementation< 8,    8,    4,    48,   512,   438>,
        block_fft_implementation< 4,    4,    8,    27,   256,   439>,
        block_fft_implementation<16,   16,    2,    11,   512,   440>,
        block_fft_implementation< 2,    2,   16,    18,   512,   441>,
        block_fft_implementation< 2,    2,   16,    30,   256,   442>,
        block_fft_implementation<16,   16,    2,    64,   256,   443>
    >;
};

template<> struct block_fft_record<64, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    16,   512,   444>,
        block_fft_implementation< 4,    4,   16,    20,   512,   445>,
        block_fft_implementation< 4,    4,   16,    12,  1024,   446>,
        block_fft_implementation< 8,    8,    8,    13,  1024,   447>,
        block_fft_implementation<16,   16,    4,     6,  1024,   448>,
        block_fft_implementation<16,   16,    4,    26,   512,   449>,
        block_fft_implementation< 2,    2,   32,     7,  1024,   450>,
        block_fft_implementation< 2,    2,   32,    12,   512,   451>
    >;
};

template<> struct block_fft_record<128, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   16,     8,  1024,   452>,
        block_fft_implementation<16,   16,    8,     4,  2048,   453>,
        block_fft_implementation<16,   16,    8,    16,  1024,   454>,
        block_fft_implementation< 4,    4,   32,     5,  2048,   455>,
        block_fft_implementation< 4,    4,   32,     8,  1024,   456>,
        block_fft_implementation< 8,    8,   16,     8,  2048,   457>,
        block_fft_implementation< 2,    2,   64,     6,  2048,   458>,
        block_fft_implementation< 2,    2,   64,     6,  1024,   459>
    >;
};

template<> struct block_fft_record<256, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   32,     4,  2048,   460>,
        block_fft_implementation<16,   16,   16,     2,  4096,   461>,
        block_fft_implementation<16,   16,   16,     2,  2048,   462>,
        block_fft_implementation< 4,    4,   64,     2,  4096,   463>,
        block_fft_implementation< 4,    4,   64,     4,  2048,   464>,
        block_fft_implementation< 8,    8,   32,     4,  4096,   465>,
        block_fft_implementation< 2,    2,  128,     3,  2048,   466>,
        block_fft_implementation< 2,    2,  128,     2,  4096,   467>
    >;
};

template<> struct block_fft_record<512, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     2,  4096,   468>,
        block_fft_implementation< 8,    8,   64,     2,  8192,   469>,
        block_fft_implementation<16,   16,   32,     1,  8192,   470>,
        block_fft_implementation<16,   16,   32,     4,  4096,   471>,
        block_fft_implementation< 4,    4,  128,     2,  4096,   472>,
        block_fft_implementation< 4,    4,  128,     1,  8192,   473>,
        block_fft_implementation< 2,    2,  256,     2,  4096,   474>,
        block_fft_implementation< 2,    2,  256,     1,  8192,   475>
    >;
};

template<> struct block_fft_record<1024, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  128,     1,  8192,   476>,
        block_fft_implementation<16,   16,   64,     2,  8192,   477>,
        block_fft_implementation< 4,    4,  256,     1,  8192,   478>,
        block_fft_implementation< 4,    4,  256,     1, 16384,   479>,
        block_fft_implementation<16,   16,   64,     2, 16384,   480>,
        block_fft_implementation< 8,    8,  128,     1, 16384,   481>,
        block_fft_implementation< 2,    2,  512,     1, 16384,   482>,
        block_fft_implementation< 2,    2,  512,     1,  8192,   483>
    >;
};

template<> struct block_fft_record<2048, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  256,     1, 16384,   484>,
        block_fft_implementation<16,   16,  128,     1, 16384,   485>,
        block_fft_implementation<16,   16,  128,     1, 32768,   486>,
        block_fft_implementation< 8,    8,  256,     1, 32768,   487>,
        block_fft_implementation< 4,    4,  512,     1, 16384,   488>,
        block_fft_implementation< 4,    4,  512,     1, 32768,   489>,
        block_fft_implementation< 2,    2, 1024,     1, 16384,   490>,
        block_fft_implementation< 2,    2, 1024,     1, 32768,   491>
    >;
};

template<> struct block_fft_record<4096, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  512,     1, 32768,   492>,
        block_fft_implementation<16,   16,  256,     1, 32768,   493>,
        block_fft_implementation<16,   16,  256,     1, 65536,   494>,
        block_fft_implementation< 8,    8,  512,     1, 65536,   495>,
        block_fft_implementation< 4,    4, 1024,     1, 32768,   496>,
        block_fft_implementation< 4,    4, 1024,     1, 65536,   497>
    >;
};

template<> struct block_fft_record<8192, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  512,     1, 65536,   498>,
        block_fft_implementation< 8,    8, 1024,     1, 65536,   499>
    >;
};

template<> struct block_fft_record<3, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,    64,     0,   500>
    >;
};

template<> struct block_fft_record<9, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,    32,     0,   501>,
        block_fft_implementation< 3,    3,    3,    80,    72,   502>,
        block_fft_implementation< 3,    3,    3,    64,   144,   503>
    >;
};

template<> struct block_fft_record<27, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,   128,     0,   504>,
        block_fft_implementation< 9,    9,    3,    21,   216,   505>,
        block_fft_implementation< 9,    9,    3,    80,   432,   506>,
        block_fft_implementation< 3,    3,    9,    30,   216,   507>,
        block_fft_implementation< 3,    3,    9,    23,   432,   508>
    >;
};

template<> struct block_fft_record<81, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    14,  1296,   509>,
        block_fft_implementation< 9,    9,    9,    14,   648,   510>,
        block_fft_implementation<27,   27,    3,    10,   648,   511>,
        block_fft_implementation<27,   27,    3,     9,  1296,   512>,
        block_fft_implementation< 3,    3,   27,     8,  1296,   513>,
        block_fft_implementation< 3,    3,   27,    13,   648,   514>
    >;
};

template<> struct block_fft_record<243, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    9,    14,  1944,   515>,
        block_fft_implementation<27,   27,    9,     3,  3888,   516>,
        block_fft_implementation< 9,    9,   27,     9,  3888,   517>,
        block_fft_implementation< 9,    9,   27,     9,  1944,   518>,
        block_fft_implementation< 3,    3,   81,     3,  3888,   519>,
        block_fft_implementation< 3,    3,   81,     3,  1944,   520>
    >;
};

template<> struct block_fft_record<729, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,   81,     3, 11664,   521>,
        block_fft_implementation< 9,    9,   81,     3,  5832,   522>,
        block_fft_implementation<27,   27,   27,     1,  5832,   523>,
        block_fft_implementation<27,   27,   27,     1, 11664,   524>,
        block_fft_implementation< 3,    3,  243,     1,  5832,   525>,
        block_fft_implementation< 3,    3,  243,     1, 11664,   526>
    >;
};

template<> struct block_fft_record<2187, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  243,     1, 17496,   527>,
        block_fft_implementation< 9,    9,  243,     1, 34992,   528>,
        block_fft_implementation< 3,    3,  729,     1, 17496,   529>,
        block_fft_implementation< 3,    3,  729,     1, 34992,   530>
    >;
};

template<> struct block_fft_record<6561, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  729,     1, 52488,   531>
    >;
};

template<> struct block_fft_record<5, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    64,     0,   532>
    >;
};

template<> struct block_fft_record<25, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,   128,     0,   533>,
        block_fft_implementation< 5,    5,    5,    25,   400,   534>,
        block_fft_implementation< 5,    5,    5,    25,   200,   535>
    >;
};

template<> struct block_fft_record<125, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,   25,     5,  2000,   536>,
        block_fft_implementation< 5,    5,   25,     5,  1000,   537>,
        block_fft_implementation<25,   25,    5,    25,  1000,   538>,
        block_fft_implementation<25,   25,    5,     6,  2000,   539>
    >;
};

template<> struct block_fft_record<625, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,     5,  5000,   540>,
        block_fft_implementation< 5,    5,  125,     1,  5000,   541>,
        block_fft_implementation< 5,    5,  125,     1, 10000,   542>,
        block_fft_implementation<25,   25,   25,     5, 10000,   543>
    >;
};

template<> struct block_fft_record<3125, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,  625,     1, 50000,   544>,
        block_fft_implementation< 5,    5,  625,     1, 25000,   545>
    >;
};

template<> struct block_fft_record<7, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,    32,     0,   546>
    >;
};

template<> struct block_fft_record<49, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    18,   392,   547>,
        block_fft_implementation< 7,    7,    7,    18,   784,   548>
    >;
};

template<> struct block_fft_record<343, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,     5,  5488,   549>,
        block_fft_implementation< 7,    7,   49,     5,  2744,   550>
    >;
};

template<> struct block_fft_record<2401, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     1, 38416,   551>,
        block_fft_implementation< 7,    7,  343,     1, 19208,   552>
    >;
};

template<> struct block_fft_record<11, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,    32,     0,   553>
    >;
};

template<> struct block_fft_record<121, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    23,  1936,   554>,
        block_fft_implementation<11,   11,   11,    23,   968,   555>
    >;
};

template<> struct block_fft_record<1331, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     1, 10648,   556>,
        block_fft_implementation<11,   11,  121,     1, 21296,   557>
    >;
};

template<> struct block_fft_record<6, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    32,     0,   558>
    >;
};

template<> struct block_fft_record<36, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    21,   576,   559>,
        block_fft_implementation< 6,    6,    6,    21,   288,   560>
    >;
};

template<> struct block_fft_record<216, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,     7,  1728,   561>,
        block_fft_implementation< 6,    6,   36,     7,  3456,   562>
    >;
};

template<> struct block_fft_record<1296, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     1, 10368,   563>,
        block_fft_implementation< 6,    6,  216,     1, 20736,   564>
    >;
};

template<> struct block_fft_record<10, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    32,     0,   565>
    >;
};

template<> struct block_fft_record<100, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,     3,  1600,   566>,
        block_fft_implementation<10,   10,   10,     3,   800,   567>
    >;
};

template<> struct block_fft_record<1000, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,     5, 16000,   568>,
        block_fft_implementation<10,   10,  100,     1,  8000,   569>
    >;
};

template<> struct block_fft_record<12, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,    32,     0,   570>
    >;
};

template<> struct block_fft_record<144, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    10,  1152,   571>,
        block_fft_implementation<12,   12,   12,     5,  2304,   572>
    >;
};

template<> struct block_fft_record<1728, double, fft_type::c2c, fft_direction::forward, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     2, 13824,   573>,
        block_fft_implementation<12,   12,  144,     3, 27648,   574>
    >;
};
