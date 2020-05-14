function startReadLabeledFile() {
    //obtain input element through DOM
//    var training_data = new FormData($('form').get(0));
//    $.ajax({
//        url: '/training_data_upload_view/',
//        type: 'post',
//        data: training_data,
//        cache: false,
//        processData: false,
//        contentType: false,
//        success: function(data) {
//            if(data == 'sucess'){
////                $('#run-document-classifier').attr('disabled', false);
//            }
//        }
//    });

    //event.preventDefault();
    var file = document.getElementById('unsupervised-labeled-datafile').files[0];
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
                if(new_data && new_data.length){
                    (new_data)
                }
                var labeledA = document.getElementById('unsupervised-labeled-datafile');
                var theSplit1 = labeledA.value.split('\\');
                if (labeledA.value != '') {
                    unsupervisedLabeled.innerHTML = theSplit1[theSplit1.length - 1] + " " + "(" + new_data + ")";
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
    $('#unsupervised-labeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#unsupervised-labeled-datafile').val('');
    $('#unsupervised-model').val('');
    $('#clusters-count').val('');
    $('#top-words-count').val('');
    $('#model-parameters').val('K-means Clustering');
    $('#run-document-classifier').attr('disabled', true);
    $("input[name='learning-model']").prop('checked', false);
    $("#stopWords").val('');
    $("#errorMessageFile").hide();
    $("#errorMessageClusterCount").hide();
    $("#errorMessageWordsCount").hide();
}

$("#input-reset").click(() => {
    $('#unsupervised-labeled-datafile').next('.custom-file-label').removeClass("selected").html(
    'Choose file');
    $('#unsupervised-labeled-datafile').val('');
    $('#unsupervised-model').val('');
    $('#clusters-count').val('');
    $('#top-words-count').val('');
    $('#save-model').attr('disabled', true);
    $('#run-document-classifier').attr('disabled', true);
    $('#model-parameters').val('K-means Clustering');
    $("input[name='learning-model']").prop('checked', false);
    $("#stopWords").val('');
    $("#errorMessageFile").hide();
    $("#errorMessageClusterCount").hide();
    $("#errorMessageWordsCount").hide();
});

$('.custom-file-input').on('change', function() {
    var fileName = $(this).val().split('\\').pop();
    var fileContent = $(this).val();
});

$('#run-document-classifier').click( () => {
    $('#progress-display-section').html('');
    $('#progress-bar').css('width', 10 + '%');
    $('#topics-data-table tbody').empty();
    var labeled_file_validation = $('#unsupervised-labeled-datafile')[0].files;
    var clusters_count = $('#clusters-count').val();
    var top_words_count = $('#top-words-count').val();
    if(labeled_file_validation.length > 0 && clusters_count.length > 0 && top_words_count.length > 0){
        $("#errorMessageFile").hide();
        $("#errorMessageClusterCount").hide();
        $("#errorMessageWordsCount").hide();
        $("#myModal").modal({
            keyboard: false,
            backdrop: 'static'
        });
        const reader = new FileReader();

        var unsupervisedLabeledFileName = $('#unsupervisedLabeled').text().replace(/\(|\)/g, '').split(' ');
        var labeledFileType = $('#unsupervisedLabeled').text();
        var training_data = labeledFileType.match(/\((.*)\)/);
        var automatic_Mode_checked = $('#auto-learning-model').is(':checked')
        var training_data_type = training_data[1]
        var model = $('#unsupervised-model').find("option:selected").attr('value');
        var target_performance_measure = $('#model-parameters').find("option:selected").attr('value');
        var additional_stopwords = $('#stopWords').val();
        var automatic_mode = automatic_Mode_checked;
        var training_file_name = unsupervisedLabeledFileName[0];
        var additional_stopwords = $('#stopWords').val();
        var inputData = {
        "training_data_type": training_data_type,
        "model": model,
        "additional_stopwords": additional_stopwords,
        "training_file_name" : training_file_name,
        "number_of_top_words": top_words_count,
        "number_of_clusters": clusters_count
        }

        $.ajax({
            url: '/userRunModelTrackUSL/',
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

        fetchUpdateOutput(10000)
    }
    else {
        $("#errorMessageFile").show();
        $("#errorMessageClusterCount").show();
        $("#errorMessageWordsCount").show();
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
                if(data && data.data.final_progress_value == 200 || JSON.stringify(data).indexOf('Error running the program') > -1) {
                    clearInterval(fetchDataInterval);
                }

                var selectedModel = data.data.str_parameter_name +''+ data.data.optimal_model_parameter
                var progressBarValue = (data.data.progressbar_value/data.data.progressbar_maximum) * 100 ;
                if(data.data.progress_text) {
                    $('#progress-display-section').html(data.data.progress_text);
                }
                if(data){
                    $('#progress-bar').css('width', progressBarValue + '%')
                    if(data.data.final_progress_value == 200){
                        $('#progress-bar').css('width', 100 + '%')
                        $('#save-model').attr('disabled', false);
                        setTimeout(() => {
                            $("#myModal").modal("hide");
                        }, 5000)
                        $("#download2").show();
                        $("#buttonDownload2").show();
                    }
                    if(data.data.errorString != ''){
                        $('#processError').children('h5').text(data.data.errorString)
                        $("#myModal").modal("hide");
                        $("#endProcessError").modal({
                        keyboard: false,
                        backdrop: 'static'
                        });
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

function runClassifier(inputData){
var formData = new FormData($('form').get(0));
    formData.append('inputData', JSON.stringify(inputData));
    $.ajax({
        url: '/runDocumentClassifierUnsupervised/',
        type: "post",
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
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
    var unsupervised = 'unsupervised'
    var saveModelName = $("#save-model-name").val('');
    var target_performance_measure = $('#model-parameters').find("option:selected").attr('value');
    var modelName = 'K-means clustering'
    var input = {
        'exisitingProjectName':exisitingProjectName,
        'exisitingProjectDescription':exisitingProjectDescription,
        'existingSaveModel':existingSaveModel,
        'newSaveModel':newSaveModel,
        'saveProject': data,
        'existingModelDesc':existingModelDesc,
        'target_performance_measure':'',
        'learningType':unsupervised,
        'trainedModelName':modelName
    }
    formData.append('input', JSON.stringify(input));
    if(existingSaveModel && !newSaveModel){
        if(exisitingProjectName && exisitingProjectName.length > 0 && exisitingProjectDescription && exisitingProjectDescription.length > 0 && existingModelDesc && existingModelDesc.length > 0){
            $("#errorMessageSaveDescription").hide();
            $("#errorMessageSaveProject").hide();
            $("#errorMessageExMoDesc").hide();
            $.ajax({
                url: '/save_both_validation/',
                type: "post",
                data: JSON.stringify({'exisitingProjectName':exisitingProjectName,'existingSaveModel':existingSaveModel,'newSaveModel':newSaveModel,'existingModelDesc':existingModelDesc,'learningType':unsupervised,'trainedModelName':modelName}),
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
                                //                            $('#sucess-message').fadeIn('slow', function(){
                                //                            $('#sucess-message').delay(5000).fadeOut();
                                //                            });
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
                data: JSON.stringify({'newProjectName':newProjectName,'existingSaveModel':existingSaveModel,'newSaveModel':newSaveModel,'newModelDesc':newModelDesc,'learningType':unsupervised,'trainedModelName':modelName}),
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
                                //                            $('#sucess-message').fadeIn('slow', function(){
                                //                            $('#sucess-message').delay(5000).fadeOut();
                                //                            });
                                if(data = 'saved successfully'){
                                    $("#saveModal").modal("hide");
                                    $('#sucess-message').show();
                                    $('#save-model').attr('disabled', true);
                                    $('#run-document-classifier').attr('disabled', true);
                                }
                            }
                        });
                    } else {
                        $("#saveModalNewError").modal({
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
    $.ajax({
        url: '/retrieve_existing_Project_name/',
        type: "post",
        data: JSON.stringify({
            'learningType': 'unsupervised'
        }),
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

$( document ).ready(function() {
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
                    if(data.data.progressbar_maximum > 0 && data.data.progressbar_value > 0 && data.data.current_tab == 2 && data.data.saved_project_status != 200) {
                        $('#learning-input-heading').toggleClass('collapsed');
                        $('#learning-input-body').toggleClass('hide');
                        $('#learning-output-heading').removeClass('expansion-disabled');
                        $('#learning-output-heading').toggleClass('collapsed');
                        $('#learning-output-body').toggleClass('hide');
                        $("#myModal").modal({
                            keyboard: false,
                            backdrop: 'static'
                        });
                        fetchUpdateOutput(500)
                    }
                }
            }
        });
    },500);
    $('#save-model').attr('disabled', true);
    $('#run-document-classifier').attr('disabled', true);
    $('#buttonDownload2').hide();
    $('#download2').hide();
    $("#errorMessageFile").hide();
    $("#errorMessageClusterCount").hide();
    $("#errorMessageWordsCount").hide();
    $("#errorMessageSaveDescription").hide();
    $("#errorMessageSaveProject").hide();
    $("#errorMessageNewDescription").hide();
    $("#errorMessageNewProject").hide();
    $("#errorMessageExMoDesc").hide();
    $("#errorMessageNewMoDesc").hide();
    //     $('#sucess-message').hide();
});

$('#closeModal').click(() => {
    $("#myModal").modal("hide");
});