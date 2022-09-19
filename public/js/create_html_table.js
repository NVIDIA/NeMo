var CreateHTMLTable = CreateHTMLTable || {};

function getLatestExistingImage(resultsDict){
  var latestVersion = null;
  var latestKey = null;
  for(let key of Object.keys(resultsDict)){
    if (key == "best"){
      continue;
    }
    var currentVersion = key.split(":")[1].split("-")[0]
    if (latestVersion == undefined || checkVersion(latestVersion,currentVersion) == -1){
      latestVersion = currentVersion
      latestKey = key;
    }
  }
  return latestKey;
}

function getBestResultFromDict(resultsDict){
  return Object.entries(resultsDict).reduce((minVal, entry) => minVal == undefined ? entry[1][1] : Math.min(minVal,entry[1][1]), undefined)
}

function checkVersion(a,b) {
  let x=a.split('.').map(e=> parseInt(e));
  let y=b.split('.').map(e=> parseInt(e));
  let z = "";

  for(i=0;i<x.length;i++) {
      if(x[i] === y[i]) {
          z+="e";
      } else
      if(x[i] > y[i]) {
          z+="m";
      } else {
          z+="l";
      }
  }
  if (!z.match(/[l|m]/g)) {
    return 0;
  } else if (z.split('e').join('')[0] == "m") {
    return 1;
  } else {
    return -1;
  }
}

CreateHTMLTable = {
    create: function (file_base_path, fileNames) {
        var $table = $("<table class='table table-striped table-condensed' id='table-container-table'></table>");
        var $htmlTableheadString = `
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th colspan=2 style=text-align:center;>TRAIN TIME (seconds)</th>
            <th>SPEEDUP(x)</th>
            <th colspan=2 style=text-align:center;>PEAK MEMORY (MiB)</th>
            <th>SPEEDUP(x)</th>
          </tr>
          <tr>
            <th>IMAGE USED</th>
            <th>RUN ON</th>
            <th>SIZE</th>
            <th>TP</th>
            <th>PP</th>
            <th>NUM NODES</th>
            <th>PRECISION</th>
            <th>AMP FLAG</th>
            <th>NUM STEPS</th>
            <th >LATEST</th>
            <th>PREV LOWEST</th>
            <th> PREV LOWEST / LATEST </th>
            <th>LATEST</th>
            <th>PREV LOWEST</th>
            <th> PREV LOWEST / LATEST </th>
          </tr>
        </thead>
        `
        var $tableHead = $($htmlTableheadString);
        $table.append($tableHead);
        var $tableBody = $("<tbody></tbody>");
        var $containerElement = $("#table-container");
        $containerElement.empty().append($table);
        for (let i = 0; i < fileNames.length; i++ ){
            var fileName = file_base_path + "/"+ fileNames[i];
            $.ajax({
              type: "GET",
              async: false,
              url: fileName,
              success: function (trainMetrics) {
                var trainTimeMetrics = trainMetrics["train_time_metrics"]
                var latestKey = null;
                if (Object.keys(trainTimeMetrics).length >= 1){
                  latestKey = getLatestExistingImage(trainTimeMetrics)
                  var latestTrainTimeResult = trainTimeMetrics[latestKey][1].toFixed(3);
                  var latestTimestamp = trainTimeMetrics[latestKey][0];
                  delete trainTimeMetrics[latestKey]
                  var previousTrainTimeBest = "--";
                  var ratioTrainTime = "--";
                  if (Object.keys(trainTimeMetrics).length > 0){
                    previousTrainTimeBest = getBestResultFromDict(trainTimeMetrics).toFixed(3);
                    ratioTrainTime = (previousTrainTimeBest/latestTrainTimeResult).toFixed(3);
                  }
                  var colorTrainTime = ratioTrainTime < 1 ? "red" : "green";
                }
                else {
                  var latestTrainTimeResult = "--"
                  var latestTimestamp = "--"
                  var previousTrainTimeBest = "--"
                  var ratioTrainTime = "--"
                  var colorTrainTime = ratioTrainTime < 1 ? "red" : "green";
                }
                var memoryUsedMetrics = trainMetrics["peak_memory_metrics"]
                if (Object.keys(memoryUsedMetrics).length >= 1){
                  if (latestKey == null){
                    latestKey = getLatestExistingImage(memoryUsedMetrics)
                    var latestTimestamp = memoryUsedMetrics[latestKey][0];
                  }
                  latestMemoryResult = memoryUsedMetrics[latestKey][1].toFixed(3);
                  delete memoryUsedMetrics[latestKey]
                  var previousMemoryBest = "--";
                  var ratioMemory = "--";
                  if (Object.keys(memoryUsedMetrics).length > 0){
                    previousMemoryBest = getBestResultFromDict(memoryUsedMetrics).toFixed(3);
                    ratioMemory = (previousMemoryBest/latestMemoryResult).toFixed(3);
                  }   
                  colorMemory = ratioMemory < 1 ? "red" : "green";
                }
                else {
                  var latestMemoryResult = "--";
                  var previousMemoryBest = "--";
                  var ratioMemory = "--";
                  var colorMemory = "green";
                }
                var $tableBodyRow = $("<tr></tr>");
                $tableBodyRow.append("<td>" + latestKey.split("*")[0] + "</td>");
                $tableBodyRow.append("<td>" + new Date(parseFloat(latestTimestamp)*1000).toDateString() + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[1] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[2].split("tp")[1] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[3].split("pp")[1] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[4].split("nodes")[0] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[5] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[7] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[8].split(".")[0].split("steps")[0] + "</td>");
                $tableBodyRow.append("<td>" + latestTrainTimeResult + "</td>");
                $tableBodyRow.append("<td>" + previousTrainTimeBest + "</td>");
                $tableBodyRow.append("<td style=color:" + colorTrainTime + ">" + ratioTrainTime + "</td>");
                $tableBodyRow.append("<td>" + latestMemoryResult + "</td>");
                $tableBodyRow.append("<td>" + previousMemoryBest + "</td>");
                $tableBodyRow.append("<td style=color:" + colorMemory + ">" + ratioMemory + "</td>");
                $tableBody.append($tableBodyRow);
              }
            })
          }
        $table.append($tableBody);
        $table.DataTable({paging: false});
    }
};
