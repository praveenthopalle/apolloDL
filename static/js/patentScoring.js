function startReadLabeledFile() {
    //obtain input element through DOM
    var training_data = new FormData($('form').get(0));
    $.ajax({
        url: '/training_data_upload_view/',
        type: 'post',
        data: training_data,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
        }
    });

    //event.preventDefault();
    var file = document.getElementById('supervised-labeled-datafile').files[0];
    if (file) {
        var reader;
        try {
            reader = new FileReader();
        } catch (e) {
            document.getElementById('output').innerHTML = "Error: seems File API is not supported on your browser";
            return;
        }
        // Read file into memory as UTF-8
        reader.readAsText(file, "UTF-8");
        // handle success and errors

        reader.onload = readLabeledDatafile;
        reader.onerror = errorHandler;
    }
}

function startReadUnlabeledFile() {
    //obtain input element through DOM
    var testing_data = new FormData($('form').get(0));
    $.ajax({
        url: '/testing_data_upload_view/',
        type: 'post',
        data: testing_data,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            if(data == 'sucess'){
                $('#similarity-check').attr('disabled', false);
            }
        }
    });

    var unlabeledFile = document.getElementById('supervised-unlabeled-datafile').files[0];
    if (unlabeledFile) {
        //getAsText(unlabeledFile);
        var reader1;
        try {
            reader1 = new FileReader();

        } catch (e) {
            document.getElementById('output').innerHTML = "Error: seems File API is not supported on your browser";
            return;
        }
        // Read file into memory as UTF-8
        reader1.readAsText(unlabeledFile, "UTF-8");
        // handle success and errors

        reader1.onload = readUnlabeledDatafile;
        reader1.onerror = errorHandler;
    }
}

function readLabeledDatafile(evt) {
    // Obtain the read file data
    var fileString = evt.target.result;
    fileString = fileString.split('\n');
    var fileStringValue = fileString[0]
    var new_data = []
    $.ajax({
        url: '/patentScoringData/',
        type: "post",
        data: fileStringValue,
        success: function(data) {
            if(data === 'Patent' || data === 'Journal'){
                new_data.push(data);
                if (new_data && new_data.length) {
                    (new_data)
                }
                var labeledA = document.getElementById('supervised-labeled-datafile');
                var theSplit1 = labeledA.value.split('\\');
                if (labeledA.value != '') {
                    supervisedLabeled.innerHTML = theSplit1[theSplit1.length - 1] + " " + "(" + new_data + ")";
                //labeledA.value = '';
                } else {
                    alert('please upload the right file')
                }
            }else{
                alert('please upload the right file')
            }
        }
    });
}

function readUnlabeledDatafile(evt) {
    // Obtain the read file data
    var fileString = evt.target.result;
    var new_data = []
    $.ajax({
        url: '/patentScoringData/',
        type: "post",
        data: fileString,
        success: function(data) {
            if(data === 'Patent' || data === 'Journal'){
                new_data.push(data);
                var labeledB = document.getElementById('supervised-unlabeled-datafile');
                var theSplit2 = labeledB.value.split('\\');

                if (labeledB.value != '') {
                    supervisedUnlabeled.innerHTML = theSplit2[theSplit2.length - 1] + " " + "(" + new_data + ")";
                //labeledB.value = '';
                } else {
                    alert('please upload the right file')
                }
            }else{
                alert('please upload the right file')
            }
        }
    });
}

function errorHandler(evt) {
    if (evt.target.error.code == evt.target.error.NOT_READABLE_ERR) {
    // The file could not be read
        document.getElementById('output').innerHTML = "Error reading file..."
    }
}

window.addEventListener('load', function load() {
    const loader = document.getElementById('loader');
    setTimeout(function() {
        loader.classList.add('fadeOut');
    }, 300);
});

$('#learning-input-heading').on('click', (e) => {
    $('#learning-input-heading').toggleClass('collapsed');
    $('#learning-input-body').toggleClass('hide');
    $('#learning-output-heading').toggleClass('collapsed');
    $('#learning-output-body').toggleClass('hide');
});

$('#learning-output-heading').click(() => {
    $('#learning-output-heading').toggleClass('collapsed');
    $('#learning-output-body').toggleClass('hide');
    $('#learning-input-heading').toggleClass('collapsed');
    $('#learning-input-body').toggleClass('hide');
});

$('.custom-file-input').on('change', function() {
    var fileName = $(this).val().split('\\').pop();
    var fileContent = $(this).val();
    //    console.log('fileContent ==>', fileContent);
    //        $(this).next('.custom-file-label').addClass("selected").html(fileName);
});

$("input[name='search-model']").click(() => {
    var selectedValue = $("input[name='search-model']:checked").val();
    if (selectedValue == 'priorart') {
        $('#priorart-keywords').removeAttr('disabled');
        $('#supervised-labeled-datafile').attr('disabled', true);
    } else if (selectedValue == 'infringe') {
        $('#supervised-labeled-datafile').removeAttr('disabled');
        $('#priorart-keywords').attr('disabled', true);
        $('#priorart-keywords').val('');
    }
});

$('#similarity-check').click(() => {
    var selectedValue = $("input[name='search-model']:checked").val();
    var unlabeledFile_file_validation = $('#supervised-unlabeled-datafile')[0].files
    var keywordValue = $('#priorart-keywords').val();

    if (selectedValue == 'priorart') {
        if(keywordValue.length > 0 && unlabeledFile_file_validation.length > 0){
            var supervisedUnLabeledFileName = $('#supervisedUnlabeled').text().replace(/\(|\)/g, '').split(' ');
            var filename_NonSamsung_Patents = supervisedUnLabeledFileName[0];
            var unLabeledFileType = $('#supervisedUnlabeled').text();
            var testing_data = unLabeledFileType.match(/\((.*)\)/);
            var testing_data_type = testing_data[1]
            $("#myModal").modal("show")

            var inputData = {
                "filename_NonSamsung_Patents": filename_NonSamsung_Patents,
                "training_data_type": 'null',
                "testing_data_type": testing_data_type,
                "keywords": keywordValue,
                "searchType": 'keywords'
            }
            $.ajax({
                url: '/userRunModelTrackPS/',
                type: "post",
                data: JSON.stringify(inputData),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                }
            });

            $.ajax({
                url: '/patentScoringGlobals/',
                type: "post",
                data: JSON.stringify({}),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data = 'success'){
                        setTimeout(() => {runClassifier(inputData)}, 1000);
                    }
                }
            });
            fetchUpdateOutput();
        } else {
            $("#errorMessageFile").show();
            $("#errorMessageFile2").show();
        }
    } else if (selectedValue == 'infringe') {
        var labeled_file_validation = $('#supervised-labeled-datafile')[0].files;
        if(labeled_file_validation.length > 0 && unlabeledFile_file_validation.length > 0){
            $("#errorMessageFile").hide();
            $("#errorMessageFile2").hide();
            $("#myModal").modal("show")
            const reader = new FileReader();

            var supervisedLabeledFileName = $('#supervisedLabeled').text().replace(/\(|\)/g, '').split(' ');
            var supervisedUnLabeledFileName = $('#supervisedUnlabeled').text().replace(/\(|\)/g, '').split(' ');
            var unLabeledFileType = $('#supervisedUnlabeled').text();
            var testing_data = unLabeledFileType.match(/\((.*)\)/);
            var labeledFileType = $('#supervisedLabeled').text();
            var training_data = labeledFileType.match(/\((.*)\)/);
            var training_data_type = training_data[1]
            var testing_data_type = testing_data[1]
            var filename_Samsung_Patents = supervisedLabeledFileName[0];
            var filename_NonSamsung_Patents = supervisedUnLabeledFileName[0];

            var inputData = {
            "filename_NonSamsung_Patents": filename_NonSamsung_Patents,
            "training_data_type": training_data_type,
            "testing_data_type": testing_data_type,
            "keywords": 'null',
            "searchType": 'Infringement'
            }

            $.ajax({
                url: '/userRunModelTrackPS/',
                type: "post",
                data: JSON.stringify(inputData),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                }
            });

            $.ajax({
                url: '/patentScoringGlobals/',
                type: "post",
                data: JSON.stringify({}),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data = 'success'){
                        setTimeout(() => {runClassifier(inputData)}, 1000);
                    }
                }
            });
            fetchUpdateOutput()
        } else {
            $("#errorMessageFile").show();
            $("#errorMessageFile2").show();
        }
    }
});

function fetchUpdateOutput(){
    var fetchDataInterval = setInterval(() => {
        $.ajax({
            url: '/fetch_update_patentscoring/',
            type: "post",
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                var progressBarValue = (data.data.progress_value / data.data.progressbar_maximum) * 100;
                if (data && data.data.final_progress_value == 200 || JSON.stringify(data).indexOf('Error running the program') > -1) {
                    clearInterval(fetchDataInterval);
                }
                if(data && data.data.progress_text) {
                    console.log('entered to data')
                    $('#progress-display-section').html(data.data.progress_text);
                }
                if(data && data.data.progress_value > 0){
                    $('#progress-bar').css('width', (progressBarValue) + '%')
                }
                if(data && data.data.final_progress_value === 200){
                    $('#progress-bar').css('width', (100) + '%')
                    setTimeout(() => {
                        $("#myModal").modal("hide");
                    }, 5000)
                }
            }
        });
    }, 10000);
}

function runClassifier(inputData) {
    $.ajax({
        url: '/computeSimilarityBetweenSamsungAndNonSamsungPatents/',
        type: "post",
        data: JSON.stringify(inputData),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            $('#buttonDownload2').show();
            $('#download2').show();
        }
    });
}

function readFileContent(file) {
    const reader = new FileReader()
    return new Promise((resolve, reject) => {
        reader.onload = event => resolve(event.target.result)
        reader.onerror = error => reject(error)
        reader.readAsText(file, "UTF-8");
    })
}

$(document).ready(function() {
    $("#errorMessageFile").hide();
    $("#errorMessageFile2").hide();
    $('#similarity-check').attr('disabled', true);
    $('#buttonDownload2').hide();
    $('#download2').hide();
})

$('#closeModal').click(() => {
    $("#myModal").modal("hide");
});
