var video = document.getElementById('video');
//Get access to the camera!
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
// Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
//video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    });
} else if (navigator.getUserMedia) { // Standard
    navigator.getUserMedia({video: true}, function (stream) {
        video.src = stream;
        video.play();
    }, errBack);
} else if (navigator.webkitGetUserMedia) { // WebKit-prefixed
    navigator.webkitGetUserMedia({video: true}, function (stream) {
        video.src = window.webkitURL.createObjectURL(stream);
        video.play();
    }, errBack);
} else if (navigator.mozGetUserMedia) { // Mozilla-prefixed
    navigator.mozGetUserMedia({video: true}, function (stream) {
        video.srcObject = stream;
        video.play();
    }, errBack);
}
var imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', handleImage, false);
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

function handleImage(e) {
    var reader = new FileReader();
    reader.onload = function (event) {
        var img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            var c = document.getElementById("canvas");
            var ctx = c.getContext("2d");
            ctx.drawImage(img, 0, 0);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);
}

// Elements for taking the snapshot
// Trigger photo take
document.getElementById("snap").addEventListener("click", function () {
    var c = document.getElementById("canvas");
    var ctx = c.getContext("2d");
    var img = document.getElementById("video");
    ctx.drawImage(img, 0, 0,270,200);
});

var imageSaver = document.getElementById('imageSaver');

imageSaver.addEventListener('click', saveImage, false);

function saveImage(e) {
    console.log('hello');
    // var dataURL = canvas.toDataURL();
    var dataURL = canvas.toDataURL('image/png');
    $.ajax({
        type: "GET",
        url: "/save-image",
        data: {
            imgBase64: dataURL
        },
    }).done(function (o) {
        console.log('saved');
        // If you want the file to be visible in the browser
        // - please modify the callback in javascript. All you
        // need is to return the url to the file, you just saved
        // and than put the image in your browser.
    });
}