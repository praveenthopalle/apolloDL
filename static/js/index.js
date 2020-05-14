$(document).ready(function () {
    // $('.nav-item').addClass("open");
    localStorage['redirectURL'] = top.location.pathname;
    $.ajax({
        url: '/getUserName/',
        type: "post",
        data: JSON.stringify(),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            $("#userName").text(data);
            localStorage['userLoggedIn'] = data;
        }
    });

    setTimeout(function() {
        $('#switch-tab').fadeOut('show');
    }, 10000);

    var scrollables = $('.scrollable');
    if (scrollables.length > 0) {
        scrollables.each((index, el) => {
            new PerfectScrollbar(el);
        });
    }

    $('.search-toggle').on('click', e => {
        $('.search-box, .search-input').toggleClass('active');
        $('.search-input input').focus();
        e.preventDefault();
    });
    // Sidebar links
    $('.sidebar .sidebar-menu li a').on('click', function () {
        const navItems = $('.sidebar').find('.nav-item');

        navItems
        .each((index, el) => {
            $(el).removeClass('open');
        });

        $(this).parent().toggleClass("open", 1000, "slow");
    });



    // Sidebar Activity Class
    const sidebarLinks = $('.sidebar').find('.sidebar-link');

    sidebarLinks
    .each((index, el) => {
        $(el).removeClass('active');
    })
    .filter(function () {
        const href = $(this).attr('href');
        const pattern = href[0] === '/' ? href.substr(1) : href;
        return pattern === (window.location.pathname).substr(1);
    })
    .addClass('active');

    // ٍSidebar Toggle
    $('.sidebar-toggle').on('click', e => {
        $('.app').toggleClass('is-collapsed');
        e.preventDefault();
    });

     const navItems = $('.sidebar-link.active').parent().parent().parent().addClass('open');
    console.log('active navItems => ', navItems)
});

var IDLE_TIMEOUT = 90 * 60; //seconds
var _idleSecondsCounter = 0;
document.onclick = function() {
    _idleSecondsCounter = 0;
};
document.onmousemove = function() {
    _idleSecondsCounter = 0;
};
document.onkeypress = function() {
    _idleSecondsCounter = 0;
};

window.setInterval(CheckIdleTime, 1000);

function CheckIdleTime() {
    _idleSecondsCounter++;
    var oPanel = document.getElementById("SecondsUntilExpire");
    if (oPanel)
        oPanel.innerHTML = (IDLE_TIMEOUT - _idleSecondsCounter) + "";
    if (_idleSecondsCounter >= IDLE_TIMEOUT) {
        $("#timeOut").modal({
            keyboard: false,
            backdrop: 'static'
        });
    }
}

