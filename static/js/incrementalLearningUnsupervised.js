window.addEventListener('load', function load() {
    const loader = document.getElementById('loader');
    setTimeout(function() {
        loader.classList.add('fadeOut');
    }, 300);
});

function startReadUnlabeledFile() {
    //obtain input element through DOM
//    var testing_data = new FormData($('form').get(0));
//    $.ajax({
//        url: '/training_data_upload_view/',
//        type: 'post',
//        data: testing_data,
//        cache: false,
//        processData: false,
//        contentType: false,
//        success: function(data) {
//            if(data == 'sucess'){
//                $('#run-document-classifier').attr('disabled', false);
//            }
//        }
//    });

    var unlabeledFile = document.getElementById('unsupervised-unlabeled-datafile').files[0];
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

function errorHandler(evt) {

    if (evt.target.error.code == evt.target.error.NOT_READABLE_ERR) {
        // The file could not be read
        document.getElementById('output').innerHTML = "Error reading file..."
    }
}

function readUnlabeledDatafile(evt) {
    // Obtain the read file data

    var fileString = evt.target.result;
    fileString = fileString.split('\n');
    var fileStringValue = fileString[0]
    var new_data = []
    $.ajax({
        url: '/data/',
        type: "post",
        data: fileStringValue,
        success: function(data) {
            if(data === 'Patent' || data === 'Journal'){
                new_data.push(data);
                var labeledB = document.getElementById('unsupervised-unlabeled-datafile');
                var theSplit2 = labeledB.value.split('\\');

                if (labeledB.value != '') {
                    unSupervisedUnLabeled.innerHTML = theSplit2[theSplit2.length - 1] + " " + "(" + new_data + ")";
                    $('#run-document-classifier').attr('disabled', false);
                } else {
                    alert('please upload the right file')
                }
            }else{
                alert('please upload the right file')
            }
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
});

$('#run-document-classifier').click(() => {
    $('#progress-display-section').html('');
    $('#progress-bar').css('width', 10 + '%')
    var projectOptionSelected = $('#incrementalLearning').find("option:selected");
    var valueSelected  = projectOptionSelected.val();
    var textSelected   = projectOptionSelected.text();
    var modelOptionSelected = $('#incrementalModels').find("option:selected").text();
    var labeled_file_validation = $('#unsupervised-unlabeled-datafile')[0].files;

    if (textSelected.indexOf('Choose') === -1 && modelOptionSelected.indexOf('Choose') === -1 && labeled_file_validation.length > 0) {

        const reader = new FileReader();

        var unsupervisedLabeledFileName = $('#unSupervisedUnLabeled').text().replace(/\(|\)/g, '').split(' ');
        var labeledFileType = $('#unSupervisedUnLabeled').text();
        var training_data = labeledFileType.match(/\((.*)\)/);

        var training_data_type = training_data[1]
        var training_file_name = unsupervisedLabeledFileName[0];
        var saveProjectName = $('#incrementalLearning').find("option:selected").text()
        var trainedModelName = $('#incrementalModels').find("option:selected").text()
        var inputData = {
        "saveProjectName": saveProjectName,
        "trainedModelName": trainedModelName,
        "training_data_type": training_data_type,
        "training_file_name": training_file_name,
        }

        $.ajax({
            url: '/userRunModelTrackIL/',
            type: "post",
            data: JSON.stringify(inputData),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
            }
        });

        $.ajax({
            url: '/runUnsupervisedSaving/',
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
        $('#learning-input-heading').toggleClass('collapsed');
        $('#learning-input-body').toggleClass('hide');
        $('#learning-output-heading').removeClass('expansion-disabled');
        $('#learning-output-heading').toggleClass('collapsed');
        $('#learning-output-body').toggleClass('hide');

        setTimeout(() => {
            $.ajax({
                url: '/fetch_update_unsupervised/',
                type: "post",
                data: JSON.stringify({}),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if (data && data.progress_text) {
                        $('#progress-display-section').html(data.progress_text);
                    }
                }
            });
        }, 1000)
        fetchUpdateOutput(5000)
    } else {
        $("#errorMessageProjectOption").show();
        $("#errorMessageModelOption").show();
        $("#errorMessageFile").show();
    }
});

function fetchUpdateOutput(time){
    var fetchDataInterval = setInterval(() => {
        $.ajax({
            url: '/fetch_update_unsupervised/',
            type: "post",
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data && data.data.final_progress_value == 200 || JSON.stringify(data).indexOf('Error running the program') > -1) {
                    clearInterval(fetchDataInterval);
                }

                var progressBarValue = (data.data.progressbar_value / data.data.progressbar_maximum) * 100;
                if (data.data.progress_text) {
                    $('#progress-display-section').html(data.data.progress_text);
                }
                if(data.data.errorString != ''){
                    $('#processError').children('h5').text(data.data.errorString)
                    $("#myModal").modal("hide");
                    $("#saveModalNewError").modal({
                    keyboard: false,
                    backdrop: 'static'
                    });
                }
                if (data) {
                    $('#progress-bar').css('width', progressBarValue + '%')
                    if (data.data.final_progress_value == 200) {
                        $('#progress-bar').css('width', 100 + '%')
                        $('#save-model').attr('disabled', false);
                        $("#myModal").modal("hide");
                    }

                    if (data.data.clusterTopicsAndCounts && data.data.clusterTopicsAndCounts.length > 0 && data.data.final_progress_value == 200) {
                        var list = data.data.clusterTopicsAndCounts;
                        $.each(list, function(i, item) {
                            $('#topics-data-table')
                            .find('tbody')
                            .append('<tr>')
                            .append('<td>' + item[0] + '</td>')
                            .append('<td>' + item[1] + '</td>')
                            .append('</tr>');
                        });
                    }
                }
            }
        });
    }, time);
}

function runClassifier(inputData) {
    var formData = new FormData($('form').get(0));
    formData.append('inputData', JSON.stringify(inputData));
    $.ajax({
        url: '/incrementalUnsupervisedLearning/',
        type: "post",
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
        }
    });
}

$('#incrementalLearning').on('change', function (e) {
    var optionSelected = $(this).find("option:selected");
    var valueSelected  = optionSelected.val();
    var textSelected   = optionSelected.text();

    if(valueSelected) {
        $.ajax({
            url: '/retreieve_Model_for_seleted/',
            type: "post",
            data: JSON.stringify({'exisitingProjectName': textSelected,'learningType': 'unsupervised'}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                $('#incrementalModels').find('option:not(:first)').remove();
                if(data && data['hits'] && data['hits']['hits']) {
                    var results = data['hits']['hits'].map(function(i) {
                        return i['_source']['model_data']['trainedModelName']
                    });

                    $('#incrementalModels').on('change', function (e) {
                        var optionSelected = $(this).find("option:selected");
                        var valueSelected  = optionSelected.val();
                        var textSelected   = optionSelected.text();
                        var trainingData = data['hits']['hits'][valueSelected]['_source']['model_data'];
                        $('#training-examples').val(trainingData.trainingDataNumInstances)
                        $('#training-classes').val(trainingData.trainingDataNumClasses)

                        var trainingTableData = trainingData.clusterTopicsAndCounts.substring(1, trainingData.clusterTopicsAndCounts.length-1);
                        trainingTableData = trainingTableData.replace(/\], \[/g, ']@@@[');
                        var arrTrainingData = trainingTableData.split('@@@');
                        if (arrTrainingData && arrTrainingData.length > 0) {
                            var list = arrTrainingData;
                            $.each(list, function(i, item) {
                                var rowData = JSON.parse(item);
                                $('#topics-data-table')
                                .find('tbody')
                                .append('<tr>')
                                .append('<td>' + rowData[0] + '</td>')
                                .append('<td>' + rowData[1] + '</td>')
                                .append('</tr>');
                            });
                        }
                    });

                    $('#project-description').val(data['hits']['hits'][0]['_source']['save_project_description']);

                    if(results && results.length > 0) {
                        $.each(results, function(key, value) {
                            $('#incrementalModels')
                            .append($("<option></option>")
                            .attr("value", key)
                            .text(value));
                        });
                    }
                }
            }
        });
    }
});

$('#save-error-ok').click(() => {
    $("#saveModalNewError").modal("hide");
    $('#learning-input-heading').removeClass('expansion-disabled');
    $('#learning-input-heading').toggleClass('collapsed');
    $('#learning-input-body').toggleClass('hide');
    $('#learning-output-heading').toggleClass('collapsed');
    $('#learning-output-body').toggleClass('hide');
    $('#unSupervisedUnLabeled').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#unsupervised-unlabeled-datafile').val('');
});

$(document).ready(function() {

    var unsupervised = 'unsupervised'
    $.ajax({
        url: '/retrieve_existing_Project_name/',
        type: "post",
        data: JSON.stringify({'learningType': unsupervised}),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            var results = data['hits']['hits'].map(function(i) {
                return i['_source']['saveProjectName']
            });

            var unique = [...new Set(results)];

            $.each(unique, function(key, value) {
                $('#incrementalLearning')
                .append($("<option></option>")
                .attr("value", key)
                .text(value));
            });
        }
    });
    setTimeout(() => {
        var userName = localStorage['userLoggedIn']
        $.ajax({
            url: '/fetch_update_unsupervised/',
            type: "post",
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                if(data.toString().indexOf('list index out of range') < 0){
                    if(data.data.progressbar_maximum > 0 && data.data.current_tab == 6) {
                        $('#learning-input-heading').toggleClass('collapsed');
                        $('#learning-input-body').toggleClass('hide');
                        $('#learning-output-heading').removeClass('expansion-disabled');
                        $('#learning-output-heading').toggleClass('collapsed');
                        $('#learning-output-body').toggleClass('hide');
                        fetchUpdateOutput(500)
                    }

                    if (data && data.data.progress_text) {
                        $('#progress-display-section').html(data.progress_text);
                    }
                }
            }
        });
    },500)
    $('#progress-display-section').html('');
    $('#progress-bar').css('width', 10 + '%');
    $('#run-document-classifier').attr('disabled', true);
    $("#errorMessageProjectOption").hide();
    $("#errorMessageModelOption").hide();
    $("#errorMessageFile").hide();
});
