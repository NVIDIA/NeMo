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

    // Hide righr nav sidebar 
    var sideNavContent = document.getElementById("bd-toc-nav");
    if(sideNavContent != null) {
        var sideBar = sideNavContent.parentElement;
        if(sideBar != null) {
            sideBar.classList.remove('show');
        }
    }
});