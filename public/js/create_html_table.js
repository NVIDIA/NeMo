var CreateHTMLTable = CreateHTMLTable || {};

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
            <th>PREVIOUS</th>
            <th> PREVIOUS / LATEST </th>
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
              success: function (trainTime) {
                var latest = null;
                var latestKey = null;
                for(let key of Object.keys(trainTime)){
                  var timeStamp = key.split("*")[1]
                  if (latest == undefined || latest < timeStamp){
                    latest = timeStamp
                    latestKey = key;
                  }
                }
                var latestResult= trainTime[latestKey];
                var previousBestResult = trainTime["best"];
                var ratio = previousBestResult/latestResult;
                let color = ratio < 1 ? "red" : "green";
                console.log(latest)
                var $tableBodyRow = $("<tr></tr>");
                $tableBodyRow.append("<td>" + latestKey.split("*")[0] + "</td>");
                $tableBodyRow.append("<td>" + new Date(parseInt(latest)) + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[1] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[2] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[3] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[4] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[5] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[7] + "</td>");
                $tableBodyRow.append("<td>" + fileName.split("_")[8].split(".")[0] + "</td>");
                $tableBodyRow.append("<td>" + latestResult.toFixed(3) + "</td>");
                $tableBodyRow.append("<td>" + previousBestResult.toFixed(3) + "</td>");
                $tableBodyRow.append("<td style=color:" + color + ">" + ratio.toFixed(3) + "</td>");
                $tableBody.append($tableBodyRow);
              }
            })
          }
        $table.append($tableBody);
    }
};
