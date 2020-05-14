function startReadLabeledFile() {
//obtain input element through DOM
//    var training_data = new FormData($('#training').get(0));
////    var training = $('#training').attr('action');
//    console.log('training_data',training_data)
//    $.ajax({
//        url: '/training_data_upload_view/',
//        type: 'post',
//        data: training_data,
//        cache: false,
//        processData: false,
//        contentType: false,
//        success: function(data) {
//        }
//    });

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
//    var testing_data = new FormData($('#testing').get(0));
//    $.ajax({
//        url: '/testing_data_upload_view/',
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

    var unlabeledFile = document.getElementById('supervised-unlabeled-datafile').files[0];
    if (unlabeledFile) {
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
        url: '/data/',
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
                } else {
                    alert("The input file header does not match either the patents data format or the journal data format. Please check and correct the file header and try again.\n"
                    + "The expected header for patent data is:\nColumn A: Identification Number\nColumn B: Title\nColumn C: Abstract\nColumn D: Claims\nColumn E: Application Number\nColumn F: Application Date\nColumn G: Current Assignee\nColumn H: UPC\n"
                    + "The expected header for journal data is:\nColumn A: Meta Data\nColumn B: Title\nColumn C: Abstract\nColumn D: Author\nColumn E: Affiliation\nColumn F: Published Year")
                }
            }else{
                alert("The input file header does not match either the patents data format or the journal data format. Please check and correct the file header and try again.\n"
                + "The expected header for patent data is:\nColumn A: Identification Number\nColumn B: Title\nColumn C: Abstract\nColumn D: Claims\nColumn E: Application Number\nColumn F: Application Date\nColumn G: Current Assignee\nColumn H: UPC\n"
                + "The expected header for journal data is:\nColumn A: Meta Data\nColumn B: Title\nColumn C: Abstract\nColumn D: Author\nColumn E: Affiliation\nColumn F: Published Year")
            }
        }
    });
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
                var labeledB = document.getElementById('supervised-unlabeled-datafile');
                var theSplit2 = labeledB.value.split('\\');

                if (labeledB.value != '') {
                    supervisedUnlabeled.innerHTML = theSplit2[theSplit2.length - 1] + " " + "(" + new_data + ")";
                    $('#run-document-classifier').attr('disabled', false);
                //labeledB.value = '';
                } else {
                    alert("The input file header does not match either the patents data format or the journal data format. Please check and correct the file header and try again.\n"
                    + "The expected header for patent data is:\nColumn A: Identification Number\nColumn B: Title\nColumn C: Abstract\nColumn D: Claims\nColumn E: Application Number\nColumn F: Application Date\nColumn G: Current Assignee\nColumn H: UPC\n"
                    + "The expected header for journal data is:\nColumn A: Meta Data\nColumn B: Title\nColumn C: Abstract\nColumn D: Author\nColumn E: Affiliation\nColumn F: Published Year")
                }
            }else{
                alert("The input file header does not match either the patents data format or the journal data format. Please check and correct the file header and try again.\n"
                + "The expected header for patent data is:\nColumn A: Identification Number\nColumn B: Title\nColumn C: Abstract\nColumn D: Claims\nColumn E: Application Number\nColumn F: Application Date\nColumn G: Current Assignee\nColumn H: UPC\n"
                + "The expected header for journal data is:\nColumn A: Meta Data\nColumn B: Title\nColumn C: Abstract\nColumn D: Author\nColumn E: Affiliation\nColumn F: Published Year")
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
    resetHeadingInput()
});

$('#learning-output-heading').click(() => {
    $('#learning-output-heading').toggleClass('collapsed');
    $('#learning-output-body').toggleClass('hide');
    $('#learning-input-heading').toggleClass('collapsed');
    $('#learning-input-body').toggleClass('hide');
    resetHeadingInput()
});

function resetHeadingInput(){
    $('#supervised-labeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-unlabeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-labeled-datafile').val('');
    $('#supervised-unlabeled-datafile').val('');
    $('#model').val('');
    $('#model').attr('disabled', true);
    $('#model-parameters').val('accuracy');
    $('#run-document-classifier').attr('disabled', true);
    $("input[name='learning-model']").prop('checked', false);
    $("#stopWords").val('');
    $("#errorMessageFile").hide();
    $("#errorMessageFile2").hide();
    $("#errorMessageModel").hide();
}

$("#input-reset").click(() => {
    $('#supervised-labeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-unlabeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-labeled-datafile').val('');
    $('#supervised-unlabeled-datafile').val('');
    $('#save-model').attr('disabled', true);
    $('#run-document-classifier').attr('disabled', true);
    $('#model').val('');
    $('#model').attr('disabled', true);
    $('#model-parameters').val('accuracy');
    $("input[name='learning-model']").prop('checked', false);
    $("#stopWords").val('');
    $("#errorMessageFile").hide();
    $("#errorMessageFile2").hide();
    $("#errorMessageModel").hide();
});

$("input[name='learning-model']").click(() => {
    var selectedValue = $("input[name='learning-model']:checked").val();
    if (selectedValue == 'custom') {
        $('#model').removeAttr('disabled');
    } else if (selectedValue == 'automatic') {
        $('#model').attr('disabled', true);
        $('#model').val('');
    }
});

$('.custom-file-input').on('change', function() {
    var fileName = $(this).val().split('\\').pop();
    var fileContent = $(this).val();
});


$('#save-ok').click(() => {
    var formData = new FormData($('form').get(0));
    var exisitingProjectName = $("#exisiting-project-name").find("option:selected").text();
    var exisitingProjectDescription = $("#exisiting-project-description").val();
    var newProjectName = $("#new-project-name").val();
    var newProjectDescription = $("#new-project-description").val();
    var existingSaveModel = $("#existing-project").prop("checked");
    var newSaveModel = $("#new-project").prop("checked");
    var existingModelDesc = $("#existing-model-desc").val();
    var newModelDesc = $("#new-model-desc").val();
    var supervised = 'supervised'
    var modelName = $('#selected-model').val();
    var target_performance_measure = $('#model-parameters').find("option:selected").attr('value');
    var input = {
        'exisitingProjectName':exisitingProjectName,
        'exisitingProjectDescription':exisitingProjectDescription,
        'existingSaveModel':existingSaveModel,
        'newSaveModel':newSaveModel,
        'saveProject': data,
        'existingModelDesc':existingModelDesc,
        'target_performance_measure':target_performance_measure,
        'learningType':supervised,
        'trainedModelName':modelName
    }
    formData.append('input', JSON.stringify(input));
    if(existingSaveModel && !newSaveModel){
        if(exisitingProjectName && exisitingProjectName.length > 0 && exisitingProjectDescription && exisitingProjectDescription.length > 0  && existingModelDesc && existingModelDesc.length > 0){
            $("#errorMessageSaveDescription").hide();
            $("#errorMessageSaveProject").hide();
            $("#errorMessageExMoDesc").hide();

            $.ajax({
                url: '/save_both_validation/',
                type: "post",
                data: JSON.stringify({'exisitingProjectName':exisitingProjectName,'existingSaveModel':existingSaveModel,'newSaveModel':newSaveModel,'existingModelDesc':existingModelDesc,'learningType': supervised,'trainedModelName':modelName}),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data === 'True'){
                        $.ajax({
                            url: '/save_both_existing_model/',
                            type: "post",
                            data: formData,
                            cache: false,
                            processData: false,
                            contentType: false,
                            success: function(data) {
                                //                                $('#sucess-message').fadeIn('slow', function(){
                                //                                $('#sucess-message').delay(5000).fadeOut();
                                //                                });
                                if(data = 'saved successfully'){
                                    $("#saveModal").modal("hide");
                                    $('#sucess-message').show();
                                    $('#save-model').attr('disabled', true);
                                    $('#run-document-classifier').attr('disabled', true);
                                }
                            }
                        });
                    } else {
                        // errror message for Model already exist do u want me to over right
                        $("#saveModalExistingError").modal({
                            keyboard: false,
                            backdrop: 'static'
                        });
                    }
                }
            });
        } else {
            $("#errorMessageSaveDescription").show();
            $("#errorMessageSaveProject").show();
            $("#errorMessageExMoDesc").show();
        }
    } else {
        if(newProjectName && newProjectName.length > 0 && newProjectDescription && newProjectDescription.length > 0 && newModelDesc && newModelDesc.length > 0){
            $("#errorMessageNewDescription").hide();
            $("#errorMessageNewProject").hide();
            $("#errorMessageNewMoDesc").hide();
            $.ajax({
                url: '/save_both_validation/',
                type: "post",
                data: JSON.stringify({'newProjectName':newProjectName,'existingSaveModel':existingSaveModel,'newSaveModel':newSaveModel,'newModelDesc':newModelDesc,'learningType': supervised,'trainedModelName':modelName}),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data === 'True'){
                        $.ajax({
                            url: '/save_both_existing_model/',
                            type: "post",
                            data: formData,
                            cache: false,
                            processData: false,
                            contentType: false,
                            success: function(data) {
                            //                                $('#sucess-message').fadeIn('slow', function(){
                            //                                $('#sucess-message').delay(5000).fadeOut();
                            //                                })
                                if(data = 'saved successfully'){
                                    $("#saveModal").modal("hide");
                                    $('#sucess-message').show();
                                    $('#save-model').attr('disabled', true);
                                    $('#run-document-classifier').attr('disabled', true);
                                }
                            }
                        });
                    } else {
                        $("#saveModalExistingError").modal({
                            keyboard: false,
                            backdrop: 'static'
                        });

                    }
                }
            });
        } else {
            $("#errorMessageNewDescription").show();
            $("#errorMessageNewProject").show();
            $("#errorMessageNewMoDesc").show();
        }
    }
});

$('#save-model').click(() => {
    $("#saveModal").modal({
        keyboard: false,
        backdrop: 'static'
    });
    $("input[name='existing-save-model']").prop('checked', false);
    $("input[name='new-save-model']").prop('checked', false);
    $("#exisiting-project-name").val('');
    $("#exisiting-project-description").val('');
    $("#existing-model-desc").val('');
    $("#new-model-desc").val('');
    $("#new-project-name").val('');
    $("#new-project-description").val('');
    $('#exisiting-project-name').attr('disabled', true);
    $('#exisiting-project-description').attr('disabled', true);
    $('#new-project-name').attr('disabled', true);
    $('#new-project-description').attr('disabled', true);
    $("#new-model-desc").attr('disabled', true);
    $('#existing-model-desc').attr('disabled', true);
    $("#errorMessageNewDescription").hide();
    $("#errorMessageNewProject").hide();
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageSaveProject").hide();
    $("#errorMessageExMoDesc").hide();
    $("#errorMessageNewMoDesc").hide();
    var supervised = 'supervised' //remember need to change this line in unsupervised also
    $.ajax({
        url: '/retrieve_existing_Project_name/',
        type: "post",
        data: JSON.stringify({'learningType': 'supervised'}),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            $('#exisiting-project-name').find('option:not(:first)').remove();
            var results = data['hits']['hits'].map(function(i) {
                return i['_source']['saveProjectName']
            });

            var unique = [...new Set(results)];
            $.each(unique, function(key, value) {
                $('#exisiting-project-name')
                .append($("<option></option>")
                .attr("value", key)
                .text(value));
            });
            $('#exisiting-project-name').on('change', function (e) {
                var optionSelected = $(this).find("option:selected");
                var valueSelected  = optionSelected.val();
                $('#exisiting-project-description').val(data['hits']['hits'][valueSelected]['_source']['save_project_description']);
            });
        }
    });
});

$('#save-cancel').click(() => {
    $("#saveModal").modal("hide");
});

$('#save-error-ok').click(() => {
    $("#saveModalNewError").modal("hide");
});

$('#save-error-overRight').click(() => {
    $("#saveModalExistingError").modal("hide");
});
$('#save-error-cancel').click(() => {
    $("#saveModalExistingError").modal("hide");
});

$("input[name='existing-save-model']").click(() => {
    $("input[name='new-save-model']").prop('checked', false);
    $('#exisiting-project-name').attr('disabled', false);
    $('#exisiting-project-description').attr('disabled', false);
    $("#existing-model-desc").attr('disabled', false);
    $("#new-model-desc").attr('disabled', true);
    $('#new-project-name').attr('disabled', true);
    $('#new-project-description').attr('disabled', true);
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageSaveProject").hide();
    $("#errorMessageNewDescription").hide();
    $("#errorMessageNewProject").hide();
    $("#errorMessageExMoDesc").hide();
    $("#errorMessageNewMoDesc").hide();
});

$("input[name='new-save-model']").click(() => {
    $("input[name='existing-save-model']").prop('checked', false);
    $('#exisiting-project-name').attr('disabled', true);
    $('#exisiting-project-description').attr('disabled', true);
    $("#existing-model-desc").attr('disabled', true);
    $("#new-model-desc").attr('disabled', false);
    $('#new-project-name').attr('disabled', false);
    $('#new-project-description').attr('disabled', false);
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageSaveProject").hide();
    $("#errorMessageNewDescription").hide();
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageExMoDesc").hide();
    $("#errorMessageNewMoDesc").hide();
});

$('#run-document-classifier').click(() => {
    $('#progress-display-section').html('');
    $('#progress-bar').css('width', 10 + '%');
    $('#testing-data-table tbody').empty();
    $('#training-data-table tbody').empty();
    var labeled_file_validation = $('#supervised-labeled-datafile')[0].files;
    var unlabeledFile_file_validation = $('#supervised-unlabeled-datafile')[0].files
    var Learning_model = $("#custom-learning-model").prop("checked");
    var automatic_model = $("#auto-learning-model").prop("checked");
    if(labeled_file_validation.length > 0 && unlabeledFile_file_validation.length > 0 && Learning_model || automatic_model){
        $("#errorMessageFile").hide();
        $("#errorMessageFile2").hide();
        $("#errorMessageModel").hide();
        $("#myModal").modal({
            keyboard: false,
            backdrop: 'static'
        });
        const reader = new FileReader();

        var supervisedLabeledFileName = $('#supervisedLabeled').text().replace(/\(|\)/g, '').split(' ');
        var supervisedUnLabeledFileName = $('#supervisedUnlabeled').text().replace(/\(|\)/g, '').split(' ');
        var unLabeledFileType = $('#supervisedUnlabeled').text();
        var testing_data = unLabeledFileType.match(/\((.*)\)/);
        var labeledFileType = $('#supervisedLabeled').text();
        var training_data = labeledFileType.match(/\((.*)\)/);
        var automatic_Mode_checked = $('#auto-learning-model').is(':checked')

        var training_data_type = training_data[1]
        var testing_data_type = testing_data[1]
        var model = $('#model').find("option:selected").attr('value');
        var target_performance_measure = $('#model-parameters').find("option:selected").attr('value');
        var additional_stopwords = $('#stopWords').val();
        var automatic_mode = automatic_Mode_checked;
        var training_file_name = supervisedLabeledFileName[0];
        var testing_file_name = supervisedUnLabeledFileName[0];

        var inputData = {
        "training_data_type": training_data_type,
        "testing_data_type": testing_data_type,
        "model": model,
        "target_performance_measure": target_performance_measure,
        "additional_stopwords": additional_stopwords,
        "automatic_mode": automatic_mode,
        "testing_file_name": testing_file_name,
        "training_file_name": training_file_name
        }

        $.ajax({
            url: '/userRunModelTrack/',
            type: "post",
            data: JSON.stringify(inputData),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
            }
        });

        $.ajax({
            url: '/runSupervisedSaving/',
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

        fetchUpdateOutput(10000)
    } else {
        $("#errorMessageFile").show();
        $("#errorMessageFile2").show();
        $("#errorMessageModel").show();
    }
});

function fetchUpdateOutput(time){
    var fetchDataInterval = setInterval(() => {
        $.ajax({
            url: '/fetch_update/',
            type: "post",
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data && data.data.final_progress_value == 200 || JSON.stringify(data).indexOf('Error running the program') > -1) {
                    clearInterval(fetchDataInterval);
                }

                var selectedModel = data.data.str_parameter_name + '' + data.data.optimal_model_parameter
                var progressBarValue = (data.data.progressbar_value / data.data.progressbar_maximum) * 100;
                if (data) {
                    $('#training-examples').val(data.data.trainingDataNumInstances)
                    $('#testing-examples').val(data.data.testingDataNumInstances)
                    $('#classes').val(data.data.trainingDataNumClasses)
                    $('#selected-model').val(data.data.trainedModelName)
                    $('#hyper-parameter').val(selectedModel)
                    if(data.data.progress_text) {
                        $('#progress-display-section').html(data.data.progress_text);
                    }
                    if(data.data.errorString != ''){
                        $('#processError').children('h5').text(data.data.errorString)
                        $("#myModal").modal("hide");
                        $("#endProcessError").modal({
                            keyboard: false,
                            backdrop: 'static'
                        });
                    }
                    $('#progress-bar').css('width', (progressBarValue * 2) + '%')
                    if (data.data.final_progress_value == 200) {
                        $('#progress-bar').css('width', 100 + '%')
                        $('#save-model').attr('disabled', false);
                        setTimeout(() => {
                            $("#myModal").modal("hide");
                        }, 5000)
                        $("#buttonDownload1").show();
                        $("#download1").show();
                    }
                    if (data.data.trainingDataPerformance && data.data.trainingDataPerformancesStandardDeviations) {
                        $('#average_accuracy1').val((data.data.trainingDataPerformance[0].toFixed(2)) * 100 + '%')
                        $('#average_accuracy2').val((data.data.trainingDataPerformancesStandardDeviations[0].toFixed(2)) * 100 + '%')
                        $('#average_auc1').val((data.data.trainingDataPerformance[1].toFixed(2)) * 100 + '%')
                        $('#average_auc2').val((data.data.trainingDataPerformancesStandardDeviations[1].toFixed(2)) * 100 + '%')
                        $('#average_macro_precision1').val((data.data.trainingDataPerformance[2].toFixed(2)) * 100 + '%')
                        $('#average_macro_precision2').val((data.data.trainingDataPerformancesStandardDeviations[2].toFixed(2)) * 100 + '%')
                        $('#average_macro_Recall1').val((data.data.trainingDataPerformance[3].toFixed(2)) * 100 + '%')
                        $('#average_macro_Recall2').val((data.data.trainingDataPerformancesStandardDeviations[3].toFixed(2)) * 100 + '%')
                        $('#average_macro_f11').val((data.data.trainingDataPerformance[4].toFixed(2)) * 100 + '%')
                        $('#average_macro_f12').val((data.data.trainingDataPerformancesStandardDeviations[4].toFixed(2)) * 100 + '%')
                    }
                    if(data.data.excel_status_code == 200 && data.data.final_progress_value == 200){
                        $("#download2").show();
                        $("#buttonDownload2").show();
                        $("#download3").show();
                        $("#buttonDownload3").show();
                    }

                    if (data.data.testingDataStatistics && data.data.testingDataStatistics.length > 0 && data.data.final_progress_value == 200) {
                        var list = JSON.parse(data.data.testingDataStatistics);
                        $.each(list, function(i, item) {
                            $('#testing-data-table')
                            .find('tbody')
                            .append('<tr>')
                            .append('<td>' + item[0] + '</td>')
                            .append('<td>' + item[1] + '</td>')
                            .append('<td>' + item[2] + '</td>')
                            .append('</tr>');
                        });
                    }

                    if (data.data.trainingDataStatistics && data.data.trainingDataStatistics.length > 0 && data.data.final_progress_value == 200) {
                        var list = JSON.parse(data.data.trainingDataStatistics);
                        $.each(list, function(i, item) {
                            $('#training-data-table')
                            .find('tbody')
                            .append('<tr>')
                            .append('<td>' + item[0] + '</td>')
                            .append('<td>' + item[1] + '</td>')
                            .append('<td>' + item[2] + '</td>')
                            .append('</tr>');
                        });
                    }
                }
            }
        });
    }, time);
}

function runClassifier(inputData) {
    var formData = new FormData($('#training').get(0));
    var testFile = document.getElementById('supervised-unlabeled-datafile');
    testing_data = testFile.files[0]
    formData.append('testFile', testing_data);
    formData.append('inputData', JSON.stringify(inputData));
    $.ajax({
        url: '/runDocumentClassifierSupervised/',
        type: "post",
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
        }
    })
}
function readFileContent(file) {
    const reader = new FileReader()
    return new Promise((resolve, reject) => {
        reader.onload = event => resolve(event.target.result)
        reader.onerror = error => reject(error)
        reader.readAsText(file, "UTF-8");
    })
}

$('#process-ok').click(() => {
    $("#endProcessError").modal("hide");
    $('#learning-input-heading').removeClass('expansion-disabled');
    $('#learning-input-heading').toggleClass('collapsed');
    $('#learning-input-body').toggleClass('hide');
    $('#learning-output-heading').toggleClass('collapsed');
    $('#learning-output-body').toggleClass('hide');
    $('#supervisedLabeled').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-labeled-datafile').val('');
    $('#supervisedUnlabeled').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#supervised-unlabeled-datafile').val('');
});

$(document).ready(function() {
    setTimeout(() => {
        var userName = localStorage['userLoggedIn']
        $.ajax({
            url: '/fetch_update/',
            type: "post",
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                if(data.toString().indexOf('list index out of range') < 0){
                    if(data.data.progressbar_maximum > 0 && data.data.progressbar_value > 0 && data.data.current_tab == 1 && data.data.saved_project_status != 200) {
                        $('#learning-input-heading').toggleClass('collapsed');
                        $('#learning-input-body').toggleClass('hide');
                        $('#learning-output-heading').removeClass('expansion-disabled');
                        $('#learning-output-heading').toggleClass('collapsed');
                        $('#learning-output-body').toggleClass('hide');
                        $("#myModal").modal({
                            keyboard: false,
                            backdrop: 'static'
                        });
                        fetchUpdateOutput(2000)
                    }
                }
            }
        });
    },500)
    $('#progress-display-section').html('');
    $('#progress-bar').css('width', 10 + '%');
    $('#save-model').attr('disabled', true);
    $('#run-document-classifier').attr('disabled', true);
    $("#download2").hide();
    $("#download3").hide();
    $("#download1").hide();
    $("#buttonDownload2").hide();
    $("#buttonDownload3").hide();
    $("#buttonDownload1").hide();
    $("#errorMessageFile").hide();
    $("#errorMessageFile2").hide();
    $("#errorMessageModel").hide();
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageSaveProject").hide();
    $("#errorMessageNewDescription").hide();
    $("#errorMessageExMoDesc").hide();
    $("#errorMessageNewMoDesc").hide();
    $("#errorMessageNewProject").hide();
    //     $('#sucess-message').show();
});

$('#closeModal').click(() => {
    $("#myModal").modal("hide");
});