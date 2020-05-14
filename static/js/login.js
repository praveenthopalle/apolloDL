$(document).ready(function () {
    var redirectURL = localStorage['redirectURL']
    $.ajax({
        url: '/redirectChange/',
        type: "post",
        data: JSON.stringify({'redirectURL': redirectURL}),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
        }
    });
});