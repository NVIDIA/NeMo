//Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    32,     0,   575>
    >;
};

template<> struct block_fft_record<14, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    32,     0,   576>
    >;
};

template<> struct block_fft_record<15, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    32,     0,   577>
    >;
};

template<> struct block_fft_record<17, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    32,     0,   578>
    >;
};

template<> struct block_fft_record<18, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    32,     0,   579>
    >;
};

template<> struct block_fft_record<19, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    32,     0,   580>
    >;
};

template<> struct block_fft_record<20, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    32,     0,   581>
    >;
};

template<> struct block_fft_record<21, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    32,     0,   582>
    >;
};

template<> struct block_fft_record<22, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    32,     0,   583>
    >;
};

template<> struct block_fft_record<23, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    32,     0,   584>
    >;
};

template<> struct block_fft_record<24, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    32,     0,   585>
    >;
};

template<> struct block_fft_record<26, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    32,     0,   586>
    >;
};

template<> struct block_fft_record<28, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    32,     0,   587>
    >;
};

template<> struct block_fft_record<29, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    32,     0,   588>
    >;
};

template<> struct block_fft_record<30, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    32,     0,   589>
    >;
};

template<> struct block_fft_record<31, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    32,     0,   590>
    >;
};

template<> struct block_fft_record<2, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,    32,     0,   591>
    >;
};

template<> struct block_fft_record<4, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,    96,     0,   592>,
        block_fft_implementation< 2,    2,    2,   128,    32,   593>,
        block_fft_implementation< 2,    2,    2,   128,    64,   594>
    >;
};

template<> struct block_fft_record<8, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    32,     0,   595>,
        block_fft_implementation< 4,    4,    2,   128,    64,   596>,
        block_fft_implementation< 2,    2,    4,   128,    64,   597>,
        block_fft_implementation< 4,    4,    2,   128,   128,   598>,
        block_fft_implementation< 2,    2,    4,   112,   128,   599>
    >;
};

template<> struct block_fft_record<16, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,    32,     0,   600>,
        block_fft_implementation< 4,    4,    4,    64,   256,   601>,
        block_fft_implementation< 4,    4,    4,    64,   128,   602>,
        block_fft_implementation< 2,    2,    8,    48,   256,   603>,
        block_fft_implementation< 8,    8,    2,   128,   128,   604>,
        block_fft_implementation< 8,    8,    2,    64,   256,   605>,
        block_fft_implementation< 2,    2,    8,    64,   128,   606>
    >;
};

template<> struct block_fft_record<32, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    4,    64,   256,   607>,
        block_fft_implementation< 4,    4,    8,    24,   512,   608>,
        block_fft_implementation< 8,    8,    4,    48,   512,   609>,
        block_fft_implementation< 4,    4,    8,    27,   256,   610>,
        block_fft_implementation<16,   16,    2,    11,   512,   611>,
        block_fft_implementation< 2,    2,   16,    18,   512,   612>,
        block_fft_implementation< 2,    2,   16,    30,   256,   613>,
        block_fft_implementation<16,   16,    2,    64,   256,   614>
    >;
};

template<> struct block_fft_record<64, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    8,    16,   512,   615>,
        block_fft_implementation< 4,    4,   16,    20,   512,   616>,
        block_fft_implementation< 4,    4,   16,    12,  1024,   617>,
        block_fft_implementation< 8,    8,    8,    13,  1024,   618>,
        block_fft_implementation<16,   16,    4,     6,  1024,   619>,
        block_fft_implementation<16,   16,    4,    26,   512,   620>,
        block_fft_implementation< 2,    2,   32,     7,  1024,   621>,
        block_fft_implementation< 2,    2,   32,    12,   512,   622>
    >;
};

template<> struct block_fft_record<128, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   16,     8,  1024,   623>,
        block_fft_implementation<16,   16,    8,     4,  2048,   624>,
        block_fft_implementation<16,   16,    8,    16,  1024,   625>,
        block_fft_implementation< 4,    4,   32,     5,  2048,   626>,
        block_fft_implementation< 4,    4,   32,     8,  1024,   627>,
        block_fft_implementation< 8,    8,   16,     8,  2048,   628>,
        block_fft_implementation< 2,    2,   64,     6,  2048,   629>,
        block_fft_implementation< 2,    2,   64,     6,  1024,   630>
    >;
};

template<> struct block_fft_record<256, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   32,     4,  2048,   631>,
        block_fft_implementation<16,   16,   16,     2,  4096,   632>,
        block_fft_implementation<16,   16,   16,     2,  2048,   633>,
        block_fft_implementation< 4,    4,   64,     2,  4096,   634>,
        block_fft_implementation< 4,    4,   64,     4,  2048,   635>,
        block_fft_implementation< 8,    8,   32,     4,  4096,   636>,
        block_fft_implementation< 2,    2,  128,     3,  2048,   637>,
        block_fft_implementation< 2,    2,  128,     2,  4096,   638>
    >;
};

template<> struct block_fft_record<512, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     2,  4096,   639>,
        block_fft_implementation< 8,    8,   64,     2,  8192,   640>,
        block_fft_implementation<16,   16,   32,     1,  8192,   641>,
        block_fft_implementation<16,   16,   32,     4,  4096,   642>,
        block_fft_implementation< 4,    4,  128,     2,  4096,   643>,
        block_fft_implementation< 4,    4,  128,     1,  8192,   644>,
        block_fft_implementation< 2,    2,  256,     2,  4096,   645>,
        block_fft_implementation< 2,    2,  256,     1,  8192,   646>
    >;
};

template<> struct block_fft_record<1024, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  128,     1,  8192,   647>,
        block_fft_implementation<16,   16,   64,     2,  8192,   648>,
        block_fft_implementation< 4,    4,  256,     1,  8192,   649>,
        block_fft_implementation< 4,    4,  256,     1, 16384,   650>,
        block_fft_implementation<16,   16,   64,     2, 16384,   651>,
        block_fft_implementation< 8,    8,  128,     1, 16384,   652>,
        block_fft_implementation< 2,    2,  512,     1, 16384,   653>,
        block_fft_implementation< 2,    2,  512,     1,  8192,   654>
    >;
};

template<> struct block_fft_record<2048, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  256,     1, 16384,   655>,
        block_fft_implementation<16,   16,  128,     1, 16384,   656>,
        block_fft_implementation<16,   16,  128,     1, 32768,   657>,
        block_fft_implementation< 8,    8,  256,     1, 32768,   658>,
        block_fft_implementation< 4,    4,  512,     1, 16384,   659>,
        block_fft_implementation< 4,    4,  512,     1, 32768,   660>,
        block_fft_implementation< 2,    2, 1024,     1, 16384,   661>,
        block_fft_implementation< 2,    2, 1024,     1, 32768,   662>
    >;
};

template<> struct block_fft_record<4096, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,  512,     1, 32768,   663>,
        block_fft_implementation<16,   16,  256,     1, 32768,   664>,
        block_fft_implementation<16,   16,  256,     1, 65536,   665>,
        block_fft_implementation< 8,    8,  512,     1, 65536,   666>,
        block_fft_implementation< 4,    4, 1024,     1, 32768,   667>,
        block_fft_implementation< 4,    4, 1024,     1, 65536,   668>
    >;
};

template<> struct block_fft_record<8192, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  512,     1, 65536,   669>,
        block_fft_implementation< 8,    8, 1024,     1, 65536,   670>
    >;
};

template<> struct block_fft_record<3, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,    64,     0,   671>
    >;
};

template<> struct block_fft_record<9, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,    32,     0,   672>,
        block_fft_implementation< 3,    3,    3,    80,    72,   673>,
        block_fft_implementation< 3,    3,    3,    64,   144,   674>
    >;
};

template<> struct block_fft_record<27, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,   128,     0,   675>,
        block_fft_implementation< 9,    9,    3,    21,   216,   676>,
        block_fft_implementation< 9,    9,    3,    80,   432,   677>,
        block_fft_implementation< 3,    3,    9,    30,   216,   678>,
        block_fft_implementation< 3,    3,    9,    23,   432,   679>
    >;
};

template<> struct block_fft_record<81, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    14,  1296,   680>,
        block_fft_implementation< 9,    9,    9,    14,   648,   681>,
        block_fft_implementation<27,   27,    3,    10,   648,   682>,
        block_fft_implementation<27,   27,    3,     9,  1296,   683>,
        block_fft_implementation< 3,    3,   27,     8,  1296,   684>,
        block_fft_implementation< 3,    3,   27,    13,   648,   685>
    >;
};

template<> struct block_fft_record<243, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    9,    14,  1944,   686>,
        block_fft_implementation<27,   27,    9,     3,  3888,   687>,
        block_fft_implementation< 9,    9,   27,     9,  3888,   688>,
        block_fft_implementation< 9,    9,   27,     9,  1944,   689>,
        block_fft_implementation< 3,    3,   81,     3,  3888,   690>,
        block_fft_implementation< 3,    3,   81,     3,  1944,   691>
    >;
};

template<> struct block_fft_record<729, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,   81,     3, 11664,   692>,
        block_fft_implementation< 9,    9,   81,     3,  5832,   693>,
        block_fft_implementation<27,   27,   27,     1,  5832,   694>,
        block_fft_implementation<27,   27,   27,     1, 11664,   695>,
        block_fft_implementation< 3,    3,  243,     1,  5832,   696>,
        block_fft_implementation< 3,    3,  243,     1, 11664,   697>
    >;
};

template<> struct block_fft_record<2187, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  243,     1, 17496,   698>,
        block_fft_implementation< 9,    9,  243,     1, 34992,   699>,
        block_fft_implementation< 3,    3,  729,     1, 17496,   700>,
        block_fft_implementation< 3,    3,  729,     1, 34992,   701>
    >;
};

template<> struct block_fft_record<6561, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  729,     1, 52488,   702>
    >;
};

template<> struct block_fft_record<5, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    64,     0,   703>
    >;
};

template<> struct block_fft_record<25, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,   128,     0,   704>,
        block_fft_implementation< 5,    5,    5,    25,   400,   705>,
        block_fft_implementation< 5,    5,    5,    25,   200,   706>
    >;
};

template<> struct block_fft_record<125, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,   25,     5,  2000,   707>,
        block_fft_implementation< 5,    5,   25,     5,  1000,   708>,
        block_fft_implementation<25,   25,    5,    25,  1000,   709>,
        block_fft_implementation<25,   25,    5,     6,  2000,   710>
    >;
};

template<> struct block_fft_record<625, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,     5,  5000,   711>,
        block_fft_implementation< 5,    5,  125,     1,  5000,   712>,
        block_fft_implementation< 5,    5,  125,     1, 10000,   713>,
        block_fft_implementation<25,   25,   25,     5, 10000,   714>
    >;
};

template<> struct block_fft_record<3125, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,  625,     1, 50000,   715>,
        block_fft_implementation< 5,    5,  625,     1, 25000,   716>
    >;
};

template<> struct block_fft_record<7, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,    32,     0,   717>
    >;
};

template<> struct block_fft_record<49, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    7,    18,   392,   718>,
        block_fft_implementation< 7,    7,    7,    18,   784,   719>
    >;
};

template<> struct block_fft_record<343, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,     5,  5488,   720>,
        block_fft_implementation< 7,    7,   49,     5,  2744,   721>
    >;
};

template<> struct block_fft_record<2401, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     1, 38416,   722>,
        block_fft_implementation< 7,    7,  343,     1, 19208,   723>
    >;
};

template<> struct block_fft_record<11, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,    32,     0,   724>
    >;
};

template<> struct block_fft_record<121, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    23,  1936,   725>,
        block_fft_implementation<11,   11,   11,    23,   968,   726>
    >;
};

template<> struct block_fft_record<1331, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     1, 10648,   727>,
        block_fft_implementation<11,   11,  121,     1, 21296,   728>
    >;
};

template<> struct block_fft_record<6, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    32,     0,   729>
    >;
};

template<> struct block_fft_record<36, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    21,   576,   730>,
        block_fft_implementation< 6,    6,    6,    21,   288,   731>
    >;
};

template<> struct block_fft_record<216, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,     7,  1728,   732>,
        block_fft_implementation< 6,    6,   36,     7,  3456,   733>
    >;
};

template<> struct block_fft_record<1296, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     1, 10368,   734>,
        block_fft_implementation< 6,    6,  216,     1, 20736,   735>
    >;
};

template<> struct block_fft_record<10, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    32,     0,   736>
    >;
};

template<> struct block_fft_record<100, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,     3,  1600,   737>,
        block_fft_implementation<10,   10,   10,     3,   800,   738>
    >;
};

template<> struct block_fft_record<1000, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,     5, 16000,   739>,
        block_fft_implementation<10,   10,  100,     1,  8000,   740>
    >;
};

template<> struct block_fft_record<12, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,    32,     0,   741>
    >;
};

template<> struct block_fft_record<144, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    10,  1152,   742>,
        block_fft_implementation<12,   12,   12,     5,  2304,   743>
    >;
};

template<> struct block_fft_record<1728, double, fft_type::c2c, fft_direction::inverse, 700> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     2, 13824,   744>,
        block_fft_implementation<12,   12,  144,     3, 27648,   745>
    >;
};
