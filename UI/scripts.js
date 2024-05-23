function previewContentImage() {
  var file = document.getElementById("content-image").files[0];
  var reader = new FileReader();
  reader.onloadend = function () {
    var uploadBox = document.querySelector(
      ".upload-box[onclick=\"document.getElementById('content-image').click();\"]"
    );
    uploadBox.style.backgroundImage = "url(" + reader.result + ")";
    uploadBox.style.backgroundSize = "cover";
    uploadBox.style.backgroundPosition = "center";
    uploadBox.querySelector("span").style.display = "none";
  };
  if (file) {
    reader.readAsDataURL(file);
  }
}

function previewStyleImage() {
  var file = document.getElementById("style-image").files[0];
  var reader = new FileReader();
  reader.onloadend = function () {
    var uploadBox = document.querySelector(
      ".upload-box[onclick=\"document.getElementById('style-image').click();\"]"
    );
    uploadBox.style.backgroundImage = "url(" + reader.result + ")";
    uploadBox.style.backgroundSize = "cover";
    uploadBox.style.backgroundPosition = "center";
    uploadBox.querySelector("span").style.display = "none";
  };
  if (file) {
    reader.readAsDataURL(file);
  }
}

document.getElementById("image-quality").addEventListener("input", function () {
  document.getElementById("quality-value").innerText = this.value;
});

document
  .getElementById("process-button")
  .addEventListener("click", function () {
    var contentImage = document.getElementById("content-image").files[0];
    var styleImage = document.getElementById("style-image").files[0];
    var numSteps = parseInt(document.getElementById("image-quality").value, 10);

    if (!contentImage || !styleImage) {
      document.getElementById("error-message").innerText =
        "Error: Both content and style images must be uploaded.";
      document.getElementById("error-section").style.display = "block";
      document.getElementById("result-section").style.display = "none";
      document.getElementById("social-share-buttons").style.display = "none";
      return;
    }

    var reader1 = new FileReader();
    var reader2 = new FileReader();

    reader1.onload = function (e) {
      var contentData = e.target.result;
      reader2.onload = function (e) {
        var styleData = e.target.result;

        // Initialize progress
        var processButton = document.getElementById("process-button");
        processButton.disabled = true;
        var progress = 0;
        var increment = Math.max(1, Math.floor(1000 / numSteps));
        var progressInterval = setInterval(updateProgress, increment);

        function updateProgress() {
          if (progress < 86) {
            progress += 1;
            processButton.innerText = `Processing ${progress}%`;
          }
        }

        eel.process_image(
          contentData,
          styleData,
          numSteps
        )(function (response) {
          clearInterval(progressInterval);

          if (response.startsWith("Error:")) {
            processButton.innerText = "PROCESS";
            processButton.disabled = false;
            document.getElementById("error-message").innerText = response;
            document.getElementById("error-section").style.display = "block";
            document.getElementById("result-section").style.display = "none";
            document.getElementById("social-share-buttons").style.display =
              "none";
          } else {
            var completeProgress = setInterval(function () {
              progress += 1;
              processButton.innerText = `Processing ${progress}%`;
              if (progress >= 100) {
                clearInterval(completeProgress);
                processButton.innerText = "PROCESS";
                processButton.disabled = false;

                var imgSrc = "data:image/png;base64," + response;
                document.getElementById("transformed-image").src = imgSrc;
                document.getElementById("result-section").style.display =
                  "block";
                document.getElementById("social-share-buttons").style.display =
                  "flex";
                document.getElementById("error-section").style.display = "none";

                // Create a blob URL to share the image on social media platforms
                var blob = base64ToBlob(response, "image/png");
                var url = URL.createObjectURL(blob);

                // Update social media share links
                document.getElementById("share-facebook").onclick =
                  function () {
                    var facebookShare =
                      "https://www.facebook.com/sharer/sharer.php?u=" +
                      encodeURIComponent(url);
                    window.open(facebookShare, "_blank");
                  };

                document.getElementById("share-twitter").onclick = function () {
                  var twitterShare =
                    "https://twitter.com/intent/tweet?text=Check out this transformed image&url=" +
                    encodeURIComponent(url);
                  window.open(twitterShare, "_blank");
                };

                document.getElementById("share-instagram").onclick =
                  function () {
                    alert(
                      "Instagram does not support direct image uploads via URL. Please save the image and upload it manually."
                    );
                  };
              }
            }, 50);
          }
        });
      };
      reader2.readAsDataURL(styleImage);
    };
    reader1.readAsDataURL(contentImage);
  });

document
  .getElementById("download-button")
  .addEventListener("click", function () {
    var imageSrc = document.getElementById("transformed-image").src;
    var link = document.createElement("a");
    link.href = imageSrc;
    link.download = "transformed_image.png";
    link.click();
  });

function base64ToBlob(base64, mime) {
  var sliceSize = 1024;
  var byteChars = atob(base64);
  var byteArrays = [];

  for (var offset = 0; offset < byteChars.length; offset += sliceSize) {
    var slice = byteChars.slice(offset, offset + sliceSize);
    var byteNumbers = new Array(slice.length);

    for (var i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    var byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: mime });
}
