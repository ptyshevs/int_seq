function postload() {
    setTimeout(showPage, 2000);
}

function showPage() {
  document.getElementById("preloader").style.display = "none";
  document.getElementById("content").style.display = "block";
}

function getNav() {
    var nav = document.getElementById('navbar');
    var btn = document.getElementById('sandwich');
    if (nav.className === 'topnav') {
        nav.className += ' responsive';
        btn.innerHTML = '&#10799;';
    }
    else {
        nav.className = 'topnav';
        btn.innerHTML = '&#9776;';
    }
}