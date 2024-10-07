# Long Sequence Performance

## LLAMA2-7B (FP8)

- The results in the table below show the pre-training performance of the LLAMA2-7B model with-CP (context parallelism) and without-CP for various input sequence lengths at FP8 precision. Detailed configurations and the achievable performance are provided for the with-CP configurations. For the without-CP configurations, the best achievable performance is reported within the given memory capacity constraint.

  - Container: [NeMo24.03.01.framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-H100

<style>
  table {
    border-collapse: collapse;
  }
  th {
    border: 1px solid;
    padding: 5px;
    text-align: center; /* Center-align all header cells */
  }
  td {
    border: 1px solid;
    padding: 5px;
  }
  th.top-border {
    border-top: 2px solid;
  }
  td.speedup {
    font-weight: bold;
  }
</style>


<table>
  <thead>
    <tr>
      <th rowspan="2" class="top-border">SeqLen (K)</th>
      <th rowspan="2" class="top-border"># of GPUs</th>
      <th rowspan="1" class="top-border">Without-CP</th>
      <th colspan="5" class="top-border">With-CP</th>
      <th rowspan="2" class="top-border">Speedup with-CP/without-CP</th>
    </tr>
    <tr>
      <th>TFLOPS / GPU</th>
      <th>TP</th>
      <th>PP</th>
      <th>DP</th>
      <th>CP</th>
      <th>TFLOPS / GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>768</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>768</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>8</td>
      <td>730</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>730</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>16</td>
      <td>660</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>660</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>32</td>
      <td>595</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>610</td>
      <td class="speedup">1.03</td>
    </tr>
    <tr>
      <td>64</td>
      <td>64</td>
      <td>534</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>574</td>
      <td class="speedup">1.07</td>
    </tr>
    <tr>
      <td>128</td>
      <td>128</td>
      <td>424</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>555</td>
      <td class="speedup">1.31</td>
    </tr>
    <tr>
      <td>256</td>
      <td>256</td>
      <td>392</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>549</td>
      <td class="speedup">1.40</td>
    </tr>
    <tr>
      <td>512</td>
      <td>512</td>
      <td>104</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>549</td>
      <td class="speedup">5.28</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>1024</td>
      <td>26.5</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>32</td>
      <td>536</td>
      <td class="speedup">20.23</td>
    </tr>
  </tbody>
</table>


### Speedup enabled by the CP
![Speedup Graph](speedup_figure.png)