document.addEventListener("DOMContentLoaded", function () {
    var params = window.location.search.substring(1).split("&").reduce(function (params, param) {
        if (!param) {
            return params;
        }

        var values = param.split("=");
        var name = values[0];
        var value = values[1];
        params[name] = value;
        return params;
    }, {});

    var form = document.getElementById("feedback-form");
    for (var name in params) {
        var input = form.querySelector("[name=" + name + "]");
        input.value = params[name];
    }
});