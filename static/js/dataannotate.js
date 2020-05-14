window.addEventListener('load', function load() {
    const loader = document.getElementById('loader');
    setTimeout(function() {
        loader.classList.add('fadeOut');
    }, 300);
});


var containerId = "content";
var CurrentSelection = {};
var selectedTexts = []; // marked keywords
var currentSelectedText = '';
var isRightMB = false;
var startTime = '';
var endTime = '';
var allParagraphs = []; // list of all paragraphs
var currentParaIndex = 1; // index of currently active document
var loggedinUserID = null;
var timerTrack = [];
var selectedIndexes = [];
var selectedHeaders  = [];

if (!window.CurrentSelection) {
    CurrentSelection = {}
}

CurrentSelection.Selector = {}

//get the current selection
CurrentSelection.Selector.getSelected = function () {
    var sel = '';
    if (window.getSelection) {
        sel = window.getSelection()
    } else if (document.getSelection) {
        sel = document.getSelection()
    } else if (document.selection) {
        sel = document.selection.createRange()
    }
    return sel
}
//function to be called on mouseup
CurrentSelection.Selector.mouseup = function (e) {

    e = e || window.event;
    isRightMB = false;

    if ("which" in e) // Gecko (Firefox), WebKit (Safari/Chrome) & Opera
        isRightMB = e.which == 3;
    else if ("button" in e) // IE, Opera
        isRightMB = e.button == 2;

    if (isRightMB) {
        return;
    }

    var st = CurrentSelection.Selector.getSelected();

    if (st && st.rangeCount == 0) {
        return;
    }
    if (document.selection && !window.getSelection) {
        var range = st;
    } else {
        var range = st.getRangeAt(0);

        if (range.endOffset - range.startOffset == 0 || range.toString().trim() == '') {
            return;
        }

        var s = window.getSelection();
        var node = s.anchorNode;

        try {
            while (range.toString().indexOf(' ') != 0) {
                range.setStart(node, (range.startOffset - 1));
            }

            range.setStart(node, range.startOffset + 1);

            do {
                range.setEnd(node, range.endOffset + 1);
            } while (range.toString().substring(range.toString().length - 1).trim() != '')
        } catch (error) {

        }

        var newNode = document.createElement("span");
        newNode.setAttribute("class", "selectedText");
        try {
            range.surroundContents(newNode);
        } catch (error) {
            if (window.getSelection) {
                window.getSelection().removeAllRanges();
            } else if (document.selection) {
                document.selection.empty();
            }
        }

        var title = newNode.innerHTML;
        newNode.setAttribute("title", title);

        if (title.indexOf('selectedText') > -1) {
            currentSelectedText = '';
            if (window.getSelection) {
                window.getSelection().removeAllRanges();
            } else if (document.selection) {
                document.selection.empty();
            }
            return;
        }

        if (title) {
            currentSelectedText = title;
        }

        if (selectedTexts.length === 0) {
            $("table tbody").html('');
        }

        if (currentSelectedText.length > 0) {
            selectedTexts.push(currentSelectedText);
            var markup = "<tr><td>" + currentSelectedText + "</td><td class='text-center' id='delete-td' ><i class='fa fa-trash-alt' style='color:red'></i></td></tr>";
            $("table tbody").append(markup);
        }

        if (selectedTexts.length === 1) {
            $('#delete-all-keywords').removeAttr('disabled');
        }

        currentSelectedText = '';

        //Remove Selection: To avoid extra text selection in IE
        if (window.getSelection) {
            window.getSelection().removeAllRanges();
        } else if (document.selection) {
            document.selection.empty();
        }

        try {
            var containerHTML = range.startContainer.innerHTML;
            console.log('containerHTML=> ', containerHTML)
            var header = range.startContainer.dataset['info'];
            selectedTexts.forEach((value, index) => {
                if(index == selectedTexts.length - 1) {
                    let eleOuterHTML = decodeHTML(value);
                    let contents = containerHTML.split(eleOuterHTML);
                    selectedIndexes.push(contents[0].length);
                    selectedHeaders.push(header);
                    console.log('selectedIndexes => ', selectedIndexes);
                    console.log('selectedHeaders => ', selectedHeaders);
                } else {
                    let eleOuterHTML = decodeHTML(value);
                    containerHTML = containerHTML.replace(eleOuterHTML, value)
                }
            });
        }
        catch(error) {

        }
    }
}

CurrentSelection.Selector.mousedown = function (e) {

    e = e || window.event;
    isRightMB = false;

    if ("which" in e) // Gecko (Firefox), WebKit (Safari/Chrome) & Opera
        isRightMB = e.which == 3;
    else if ("button" in e) // IE, Opera
        isRightMB = e.button == 2;

    if (isRightMB) {
        return;
    }

    $.each($('span.selectedText'), function (index, value) {
        if (selectedTexts.indexOf($(value).text()) == -1) {
            if (!$(value).text()) {
                $("#" + containerId).html($("#" + containerId).html().replace('<span class="selectedText" title=""></span>', $(value).text()))
            } else {
                var eleOuterHTML = decodeHTML($(value).text());
                $("#" + containerId).html($("#" + containerId).html().replace(eleOuterHTML, $(value).text()))
            }
        }
    });
}

/**
 * Run the app.
 */
//var timer = new Timer();
$(document).ready(function () {
    //$('#loadingModal').modal('show');
    sw.start();
    retreiveCategoryName()
//    timer();
    $.ajax({
        url: '/getUserName/',
        type: "post",
        data: JSON.stringify(),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            $(".userid-display-label").html(data);
            localStorage['userLoggedIn'] = data;
            readWorkInProgress();
            checkCookie();
            //$('#loadingModal').modal('hide');
        }
    });
    $('#validatedCustomFile').bind('change', getFile);
    $('#upload-btn').bind('click', uploadFile);
    $('#submit-btn').bind('click', loadNextDocument);
    $('#skip-btn').bind('click', skipToNextDocument);
    $('#reset-upload').bind('click', resetDocument);
    $('#delete-all-keywords').bind('click', deleteAllKeywords);
    $('#alert-close-icon').bind('click', hideAlert);
    $('.keywords-table-section').on("click", "#keywordsTable tr td", deleteKeyword);
//    $('#login-btn').bind('click', login);
//    $('#logout-btn').bind('click', logout);

    $("#" + containerId).bind("mouseup", CurrentSelection.Selector.mouseup);
    $("#" + containerId).bind("mousedown", CurrentSelection.Selector.mousedown);
});

function decodeHTML(value) {
    var newNode = document.createElement("span");
    newNode.setAttribute("class", "selectedText");
    newNode.setAttribute("title", value);
    newNode.textContent = value;
	return newNode.outerHTML
};

function readWorkInProgress() {
     $.ajax({
        url: '/dataWhenReload/',
        type: "post",
        data: JSON.stringify({}),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            if(data.training_data && data.training_data.length > 1) {
                allParagraphs = data.training_data;
                processData(data.training_data);
                $('#textPanel').removeClass('disabled-content');
                $('#para-1').removeClass('hide');
                $('#document-count-div').html(currentParaIndex + ' out of ' + (data.training_data.length - 1) + ' documents');

                if((data.training_data.length - 1) == 1) {
                    $('#submit-btn').html('Save');
                }
                updateTimer();
            } else {
                getDocumentReady();
            }
        }
    });
}

function hideAlert() {
    $('.alert').removeClass('show');
}

function getDocumentReady() {
    timerTrack = [];
    currentParaIndex = 1;
    $('.custom-file-label').html('Choose file...');
    $('#validatedCustomFile').val('');
    $('#count').val('');
    $("table tbody").html('');
    $('#uploadModal').modal('show');
    $('#sw-time').addClass('disabled-timer')
    $('#main-panel').removeClass('hide');
    $('#textPanel').removeClass('disabled-content');
    $('#reset-upload').removeAttr('disabled', 'true');
    $('#para-1').removeClass('hide');
    $('#next-btn').removeAttr('disabled', 'true');
    $('#delete-all-keywords').attr('disabled', 'true');
    $('#textPanel').addClass('disabled-content');
    $('#content').html('No content to display <br> Please upload file');
    $("table tbody").append("<tr><td class='text-center border-0' colspan='2'>No Keywords Selected</td></tr>");
    $('#document-count-div').html('');
//    $('#login-panel').css('height', window.innerHeight + 'px');
//    $('.bg-img').css('height', window.innerHeight + 'px');
//    $('#main-panel').addClass('hide');
}

function deleteKeyword(e) {
    if (this.id == 'delete-td') {
        var col = this.cellIndex,
            row = this.parentNode.rowIndex;
        $(this).closest("tr").remove();

        var keyword = selectedTexts[row - 1];

        var eleOuterHTML = decodeHTML(keyword);
        $("#" + containerId).html($("#" + containerId).html().replace(eleOuterHTML, keyword))

        selectedTexts.splice((row - 1), 1);
        selectedIndexes.splice((row - 1), 1);
        selectedHeaders.splice((row - 1), 1);
        console.log('selectedIndexes => ', selectedIndexes);
        console.log('selectedHeaders => ', selectedHeaders);

        if (selectedTexts.length === 0) {
            $('#delete-all-keywords').attr('disabled', 'true');
            $("table tbody").append("<tr><td class='text-center border-0' colspan='2'>No Keywords Selected</td></tr>");
        }
    }
}

function loadJSON(callback) {

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', 'logins.json', true); // Replace 'my_data' with the path to your file
    xobj.onreadystatechange = function () {
        if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
            callback(xobj.responseText);
        }
    };
    xobj.send(null);
}

function skipToNextDocument() {
    var result = confirm("Are you Sure, You really want to skip?");
    if (result) {
        // need to change
        sw.reset();

        sw.start();

        event.preventDefault();
        event.stopPropagation();

        $('#loadingModal').modal('show');
        selectedTexts = [];
        selectedHeaders = [];
        selectedIndexes = []
        timerTrack = [];

        $("table tbody").html("");
        $('#delete-all-keywords').attr('disabled', 'true');
        $("table tbody").append("<tr><td class='text-center border-0' colspan='2'>No Keywords Selected</td></tr>");
        $('#categoryName').val('');

        // delete current paragraph AJAX call

        $('#para-' + currentParaIndex).addClass('hide');

        currentParaIndex = currentParaIndex + 1;
        let data = {
            "categoryName": "skipButton",
            "timeTaken": "skip",
            "keywords": "skip",
            "index": "skip",
            "headers": "skip"
        }
        if(currentParaIndex > allParagraphs.length - 1) {
            $.ajax({
                url: '/saveAndContinue/',
                type: "post",
                data: JSON.stringify(data),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data){
                        $.ajax({
                            url: '/removeData/',
                            type: 'post',
                            data: JSON.stringify({}),
                            cache: false,
                            processData: false,
                            contentType: false,
                            success: function(data) {
                            }
                        });
                    }
                }
            });
            $('#submit-btn').html('Save & Continue');
            var success = confirm("successfully completed the annotation,Your file will be downloaded automatically!");
            if(success){
                $('#download2')[0].click()
                setTimeout(() => {
                    getDocumentReady();
                }, 3000);
            } else {
                setTimeout(() => {
                    $('#loadingModal').modal('hide');
                }, 1000);
                $('#sw-time').addClass('disabled-timer')
                $('#document-count-div').html('');
            }
        } else {
            if (currentParaIndex == allParagraphs.length - 1) {
                $('#submit-btn').html('Save');
            }

            $('#para-' + currentParaIndex).removeClass('hide');
            $('#document-count-div').html(currentParaIndex + ' out of ' + (allParagraphs.length - 1) + ' documents');

            setTimeout(() => {
                $('#loadingModal').modal('hide');
                updateTimer();
            }, 1000);

            event.preventDefault();
            event.stopPropagation();

            $.ajax({
                url: '/saveAndContinue/',
                type: "post",
                data: JSON.stringify(data),
                cache: false,
                processData: false,
                contentType: false,
                success: function(data) {
                    if(data){
                        $.ajax({
                            url: '/removeData/',
                            type: 'post',
                            data: JSON.stringify({}),
                            cache: false,
                            processData: false,
                            contentType: false,
                            success: function(data) {
                            }
                        });
                    }
                }
            });

        }
    }
}

function loadNextDocument() {
    if($('#categoryName').val() == '') {
        $('#alert-text').text('Please provide a valid Category Name.');
        $('.alert').addClass('show');
        event.preventDefault();
        event.stopPropagation();
    } else {
        $('#loadingModal').modal('show');
        sw.reset();

        sw.start();
        updateTimer();
        let difference = 0;
        let p = 0;
        timerTrack.forEach((time, i) => {
            if(timerTrack[i + 1] && (i % 2 == 0)) {
                startTime = timerTrack[i];
                endTime = timerTrack[i + 1];
                difference = difference + (endTime.getTime() - startTime.getTime());
            }
        });
        let resultInMinutes = Math.round(difference / 1000);
        download(resultInMinutes);

        selectedTexts = [];
        timerTrack = [];
        selectedHeaders = [];
        selectedIndexes = []

        $("table tbody").html("");
        $('#delete-all-keywords').attr('disabled', 'true');
        $("table tbody").append("<tr><td class='text-center border-0' colspan='2'>No Keywords Selected</td></tr>");
        $('#categoryName').val('');

        // delete current paragraph AJAX call

        $('#para-' + currentParaIndex).addClass('hide');

        currentParaIndex = currentParaIndex + 1;

        if(currentParaIndex > allParagraphs.length - 1) {

            $('#submit-btn').html('Save & Continue');
            var success = confirm("successfully completed the annotation,Your file will be downloaded automatically!");
            if(success){
                $('#download2')[0].click()
                setTimeout(() => {
                    getDocumentReady();
                }, 3000);
            } else {
                setTimeout(() => {
                    $('#loadingModal').modal('hide');
                }, 1000);
                $('#sw-time').addClass('disabled-timer')
                $('#document-count-div').html('');
            }
        } else {
            if (currentParaIndex == allParagraphs.length - 1) {
                $('#submit-btn').html('Save');
            }

            $('#para-' + currentParaIndex).removeClass('hide');
            $('#document-count-div').html(currentParaIndex + ' out of ' + (allParagraphs.length - 1) + ' documents');
            // form.classList.remove('was-validated');

            setTimeout(() => {
                $('#loadingModal').modal('hide');
                updateTimer();
            }, 1000);

            event.preventDefault();
            event.stopPropagation();
        }

    }

    // Fetch all the forms we want to apply custom Bootstrap validation styles to
//    var forms = document.getElementsByClassName('needs-validation');
//    // Loop over them and prevent submission
//    var validation = Array.prototype.filter.call(forms, function (form) {
//        if (form.id == 'category-form') {
//            console.log('form.checkValidity => ', form.checkValidity());
//            if (form.checkValidity() === false) {
//                form.classList.add('was-validated');
//                event.preventDefault();
//                event.stopPropagation();
//            } else {
//
//            }
//        }
//    });
}

function deleteAllKeywords() {
    selectedTexts.forEach(keyword => {
        var eleOuterHTML = decodeHTML(keyword);
        $("#" + containerId).html($("#" + containerId).html().replace(eleOuterHTML, keyword))
    });
    selectedTexts = [];
    selectedHeaders = [];
    selectedIndexes = []
    if (selectedTexts.length === 0) {
        $("table tbody").html("");
        $('#delete-all-keywords').attr('disabled', 'true');
        $("table tbody").append("<tr><td class='text-center border-0' colspan='2'>No Keywords Selected</td></tr>");
    }
}

function stopWatch(){
    if($('#pause-task').html() === 'Pause') {
        sw.stop()
    }else {
        sw.start()
    }
}

function updateTimer() {
    timerTrack.push(new Date());
//    timer.addEventListener('secondsUpdated', function (e) {
//    $('#chronoExample .values').html(timer.getTimeValues().toString());
//    });
    if(timerTrack.length > 1) {
        if($('#pause-task').html() === 'Pause') {
            $('#pause-task').html('Resume');
            $('#textPanel').addClass('disabled-text-panel');
            $('#keywords-table-section').addClass('disabled-text-panel');
            $('#reset-upload').attr('disabled', 'true');
            if(selectedTexts && selectedTexts.length > 0) {
                $('#delete-all-keywords').attr('disabled', 'true');
            }
        } else {
            $('#pause-task').html('Pause');
            $('#textPanel').removeClass('disabled-text-panel');
            $('#keywords-table-section').removeClass('disabled-text-panel');
            $('#reset-upload').removeAttr('disabled');
            if(selectedTexts && selectedTexts.length > 0) {
                $('#delete-all-keywords').removeAttr('disabled');
            }
        }
    } else {
        $('#pause-task').html('Pause');
        $('#textPanel').removeClass('disabled-text-panel');
        $('#keywords-table-section').removeClass('disabled-text-panel');
        $('#reset-upload').removeAttr('disabled');
        if(selectedTexts && selectedTexts.length > 0) {
            $('#delete-all-keywords').removeAttr('disabled');
        }
    }

    if(event) {
        event.preventDefault();
        event.stopPropagation();
    }
}

function uploadFile() {
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    var forms = document.getElementsByClassName('needs-validation');

    // Loop over them and prevent submission
    var validation = Array.prototype.filter.call(forms, function (form) {
        if (form.id == 'upload-form') {
            if (form.checkValidity() === false) {
                console.log('form is invalid')
                form.classList.add('was-validated');
                event.preventDefault();
                event.stopPropagation();
            } else {
                console.log('form is valid')
                form.classList.remove('was-validated');
                $('#uploadModal').modal('hide');
                $('#loadingModal').modal('show');

                var loadPercentage = $('#count').val();
                var loadRecords = Math.round(((allParagraphs.length - 1) * loadPercentage) / 100);

                var allParagraphsCopy = JSON.parse(JSON.stringify(allParagraphs));

                var headerItem = allParagraphsCopy.splice(0, 1);

                allParagraphsCopy = JSON.parse(JSON.stringify(allParagraphsCopy));

                allParagraphsCopy = allParagraphsCopy.slice(0, loadRecords);
                allParagraphsCopy = JSON.parse(JSON.stringify(allParagraphsCopy));

                allParagraphsCopy.sort(() => Math.random() - 0.5);
                allParagraphsCopy.splice(0, 0, headerItem[0]);
                allParagraphsCopy = JSON.parse(JSON.stringify(allParagraphsCopy));

                var formData = new FormData($('form').get(0));
                formData.append('shuffledArray', allParagraphsCopy.join('#@@#'));

                $.ajax({
                    url: '/customFileUpload/',
                    type: 'post',
                    data: formData,
                    cache: false,
                    processData: false,
                    contentType: false,
                    success: function(data) {
                        if(data == 'sucess'){
                            $('#count').val('');
                            allParagraphs = JSON.parse(JSON.stringify(allParagraphsCopy));
                            processData(allParagraphs);
                            $('#para-1').removeClass('hide');
                            $('#document-count-div').html(currentParaIndex + ' out of ' + (allParagraphs.length - 1) + ' documents');

                            $('#main-panel').removeClass('hide');
                            $('#textPanel').removeClass('disabled-content');
                            $('#reset-upload').removeAttr('disabled', 'true');

                            $('#next-btn').removeAttr('disabled', 'true');
                            setTimeout(() => {
                                $('#loadingModal').modal('hide');
                                updateTimer();
                            }, 1000);
                            $('#sw-time').removeClass('disabled-timer')
                            sw.reset();
                            sw.start();
                        } else {
                            var result = confirm('Error in upload file Please upload the right file')
                            if(result){
                                setTimeout(() => {
                                    getDocumentReady();
                                }, 1000);
                            } else {
                                setTimeout(() => {
                                    $('#loadingModal').modal('hide');
                                }, 1000);
                                $('#sw-time').addClass('disabled-timer')
                                $('#document-count-div').html('');
                            }
                        }
                    }
                });
            }
        }
    });

}

function getFile(event) {
    const input = event.target
    if ('files' in input && input.files.length > 0) {
        placeFileContent(input.files[0]);
    }
}

function placeFileContent(file) {
    $('.custom-file-label').html(file.name);
    readFileContent(file).then(content => {
    }).catch(error => console.log(error));
}

function readAndContinue(data) {
    if(data) {
        if(data.paragraphs) {
            allParagraphs = data.paragraphs;
        }
    }
}

function readFileContent(file) {
    // only called when file is uploaded
    const reader = new FileReader()
    return new Promise((resolve, reject) => {
        reader.onload = (event) => {
            $("#" + containerId).html('');
            const content = event.target.result;
            allParagraphs = content.split(/\r\n|\n/);
            allParagraphs = allParagraphs.filter(x => x != '');
            resolve(true);
        };

        reader.onerror = (event) => {
            reject(event.target.error.name);
        };

        reader.readAsText(file);
    });
}

function processData(allParagraphs) {
    let titles = [];
    // Reading line by line
    console.log('processData allParagraphs => ', allParagraphs)
    allParagraphs.forEach((line, i) => {

        if (i == 0) {
            titles = line.split(/\t/);
        } else {
            let tableNode = document.createElement("div");
            tableNode.id = 'para-' + i;
            tableNode.className = 'hide para-table';
            $("#" + containerId).append(tableNode);

            let values = line.split(/\t/);

            // titles.forEach((title, j) => {
            //     let childDiv = document.createElement("div");
            //     node.appendChild(childDiv);

            //     let childDivContent = document.createTextNode(title + ' : ' + values[j]);
            //     childDiv.appendChild(childDivContent);

            // });

            titles.forEach((title, j) => {

                let displayParagraphContent = false;
                let paragraphContents = [];
                let paragraphContent = '';

                let paras = [];
                paras = values[j].split(/([0-9]+[\.]{1})[ ]/);

                if(paras.length > 1) {
                    displayParagraphContent = true;
                    paras = paras.slice(1);
                    for (let index = paras.length - 1; index >= 0; index--) {
                        if(index % 2 === 0) {
                            paras.splice(index, 1);
                        }
                    }
                    for (let index = 0; index < paras.length; index++) {
                        const element = paras[index];
                        paragraphContents.push((index + 1).toString() + '. ' + element);
                    }
                    paragraphContent = paragraphContents.join('<br>')
                }

                let tableRowNode = document.createElement("div");
                tableRowNode.className = 'para-tablerow';
                tableNode.appendChild(tableRowNode);

                let tableCellNode1 = document.createElement("div");
                tableCellNode1.className = 'para-tablecell nowrap-text';
                tableCellNode1.setAttribute('data-info', title);
                tableRowNode.appendChild(tableCellNode1);

                let tableCellNode1Content = document.createTextNode(title);
                tableCellNode1.appendChild(tableCellNode1Content);

                let colanNode = document.createElement("div");
                colanNode.className = 'para-tablecell colan-text';
                colanNode.setAttribute('data-info', title);
                tableRowNode.appendChild(colanNode);

                let colanNodeContent = document.createTextNode(' : ');
                colanNode.appendChild(colanNodeContent);

                let tableCellNode2 = document.createElement("div");
                tableCellNode2.className = 'para-tablecell';
                tableCellNode2.setAttribute('data-info', title);
                tableRowNode.appendChild(tableCellNode2);

                if(displayParagraphContent) {
                    tableCellNode2.innerHTML = paragraphContent;
                } else {
                    let tableCellNode2Content = document.createTextNode(values[j]);
                    tableCellNode2.appendChild(tableCellNode2Content);
                }

            });
        }

    });
}

function download(timeTaken) {
    let categoryName = $('#categoryName').val();
    let data = {
        "categoryName": categoryName,
        "timeTaken": timeTaken,
        "keywords": selectedTexts,
        "index": selectedIndexes,
        "headers": selectedHeaders
    }
    $.ajax({
        url: '/saveAndContinue/',
        type: "post",
        data: JSON.stringify(data),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            if(data){
                $.ajax({
                    url: '/removeData/',
                    type: 'post',
                    data: JSON.stringify({}),
                    cache: false,
                    processData: false,
                    contentType: false,
                    success: function(data) {
                    }
                });
                setTimeout(() => {
                    retreiveCategoryName()
                }, 5000)
            }
        }
    });
    //send this data variable to AJAX call to save keywords of each document

    //below data will create and download text file to browser

//    let writeData = 'Category Name: ' + categoryName + '\n\n';
//    writeData = writeData + 'Total time: ' + timeTaken + ' Minutes' + '\n\n' + selectedTexts.join(',');
//    let file = new Blob([writeData], {
//        type: 'text/plain'
//    });
//    if (window.navigator.msSaveOrOpenBlob) // IE10+
//        window.navigator.msSaveOrOpenBlob(file, categoryName);
//    else { // Others
//        let a = document.createElement("a"),
//            url = URL.createObjectURL(file);
//        a.href = url;
//        a.download = categoryName;
//        document.body.appendChild(a);
//        a.click();
//        setTimeout(function () {
//            document.body.removeChild(a);
//            window.URL.revokeObjectURL(url);
//        }, 0);
//    }
}

//function login() {
//    loadJSON(function (response) {
//        // Parse JSON string into object
//        var actual_JSON = JSON.parse(response);
//        console.log('actual_JSON => ', actual_JSON);
//        const loginData = actual_JSON.loginData;
//        const loginId = $('#userid').val();
//        const loginPwd = $('#pwd').val();
//        var forms = document.getElementsByClassName('needs-validation');
//        if (loginData.filter(x => x.userid == loginId && x.password == loginPwd).length > 0) {
//            loggedinUserID = loginId;
//            // $('#main-panel').removeClass('hide');
//            $('#uploadModal').modal('show');
//            $('#login-panel').addClass('hide');
//
//            $('#validatedCustomFile').bind('change', getFile);
//
//            $(".userid-display-label").html('Welcome ' + loginId);
//            $('#userid').val('');
//            $('#pwd').val('');
//            Array.prototype.filter.call(forms, function (form) {
//                if (form.id == 'login-form') {
//                    form.classList.remove('was-validated');
//                }
//            });
//        } else {
//            if (loginId && loginPwd) {
//                $('#invalid-login').css('display', 'block');
//            }
//            Array.prototype.filter.call(forms, function (form) {
//                if (form.id == 'login-form') {
//                    form.classList.add('was-validated');
//                }
//            });
//        }
//    });
//}
//
//function logout() {
//    document.cookie = "sessionCookie=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
//    $('#login-panel').css('height', window.innerHeight + 'px');
//    $('.bg-img').css('height', window.innerHeight + 'px');
//    $('#sessionModal').modal('hide')
//    $('#main-panel').addClass('hide');
//    $('#login-panel').removeClass('hide');
//    $('#reset-upload').attr('disabled', 'true');
//    $('#submit-btn').attr('disabled', 'true');
//    $('#delete-all-keywords').attr('disabled', 'true');
//    $('#prev-btn').attr('disabled', 'true');
//    $('#next-btn').attr('disabled', 'true');
//    $('#textPanel').addClass('disabled-content');
//    $('#content').html('No content to display <br> Please upload file');
//    $("#main-panel").unbind();
//}

function setCookie(userid, minutes) {
    var d = new Date();
    d.setTime(d.getTime() + (minutes * 60 * 1000));
    var expires = "expires=" + d.toGMTString();
    console.log('expires => ', expires);
    var sessionCookie = {
        'userid': userid,
        'expires': d.toGMTString()
    }
    document.cookie = "sessionCookie=" + JSON.stringify(sessionCookie) + ";" + expires + ";path=/";
}

function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "{}";
}

function checkCookie() {
    try {
        var sessionCookie = JSON.parse(getCookie("sessionCookie"));
        if (sessionCookie && sessionCookie.userid) {
            // $('#main-panel').removeClass('hide');
            // $('#login-panel').addClass('hide');
            // document.getElementById("category-form").reset();
            selectedTexts = [];
            selectedHeaders = [];
            selectedIndexes = []
            $("#" + containerId).html('');
            $(".userid-display-label").html('Welcome ' + sessionCookie.userid);
    //        setSessionTimeout();
        }
    }
    catch(err) {
        console.log('exception occurred => ', err)
    }

}

function resetDocument() {
    var result = confirm("Reset document will lose the track of whole file upload, Do you want to reset Document?");
    if (result) {
        $('#loadingModal').modal('show');
        $.ajax({
            url: '/resetDocument/',
            type: 'post',
            data: JSON.stringify({}),
            cache: false,
            processData: false,
            contentType: false,
            success: function(data) {
                getDocumentReady();
                $('#loadingModal').modal('hide');
            }
        });
    }
}

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

function retreiveCategoryName(){
    $.ajax({
        url: '/categoryName/',
        type: 'post',
        data: JSON.stringify({}),
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
            if(data && data['hits'] && data['hits']['hits'] && data['hits']['hits'].length > 0){
                $('#hosting-plan').find('option:not(:first)').remove();

                var results = data['hits']['hits'].map(function(i) {
                    return i['_source']['categoryName']
                });

                var unique = [...new Set(results)];

                $.each(unique, function(key, value) {
                    $('#hosting-plan')
                    .append($("<option></option>")
                    .attr("value", value)
                    .text(value));
                });
            }
        }
    });
}





var sw = {
  /* [INIT] */
  etime : null, // holds HTML time display
  erst : null, // holds HTML reset button
  ego : null, // holds HTML start/stop button
  timer : null, // timer object
  now : 0, // current timer
  init : function () {
    // Get HTML elements
    sw.etime = document.getElementById("sw-time");

  },

  /* [ACTIONS] */
  tick : function () {
  // tick() : update display if stopwatch running

    // Calculate hours, mins, seconds
    sw.now++;
    var remain = sw.now;
    var hours = Math.floor(remain / 3600);
    remain -= hours * 3600;
    var mins = Math.floor(remain / 60);
    remain -= mins * 60;
    var secs = remain;

    // Update the display timer
    if (hours<10) { hours = "0" + hours; }
    if (mins<10) { mins = "0" + mins; }
    if (secs<10) { secs = "0" + secs; }
    sw.etime.innerHTML = hours + ":" + mins + ":" + secs;
  },

  start : function () {
  // start() : start the stopwatch

    sw.timer = setInterval(sw.tick, 1000);
  },

  stop  : function () {
  // stop() : stop the stopwatch

    clearInterval(sw.timer);
    sw.timer = null;
  },

  reset : function () {
  // reset() : reset the stopwatch

    // Stop if running
    if (sw.timer != null) { sw.stop(); }

    // Reset time
    sw.now = -1;
    sw.tick();
  }
};

window.addEventListener("load", sw.init);