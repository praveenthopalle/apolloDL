window.addEventListener('load', function load() {
    const loader = document.getElementById('loader');
    setTimeout(function() {
        loader.classList.add('fadeOut');
    }, 300);
});

function startReadLabeledFile() {
//obtain input element through DOM
    var training_data = new FormData($('form').get(0));
    $.ajax({
        url: '/Spotlight_Process_All_Files/',
        type: 'post',
        data: training_data,
        cache: false,
        processData: false,
        contentType: false,
        success: function(data) {
        console.log(data)
            topPapers(data)
            topPatents(data)
            topPatentInstitutions(data)
        }
    });

    //event.preventDefault();
    var file = document.getElementById('spotlight-datafile').files[0];
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

        reader.onerror = errorHandler;
    }
}

function errorHandler(evt) {
    if (evt.target.error.code == evt.target.error.NOT_READABLE_ERR) {
    // The file could not be read
        document.getElementById('output').innerHTML = "Error reading file..."
    }
}

function topPatents(data){
     titles = JSON.parse(data.result.topPatentTable.titles)

     const formattedDataObj = {};
     let dKeys = [];

        dKeys = Object.keys(titles);
        dKeys.forEach(ky => {
            formattedDataObj[ky]= {};
        });

     Object.keys(data.result.topPatentTable).forEach((key, i) => {
        const val = JSON.parse(data.result.topPatentTable[key]);
        console.log(key, val);

        dKeys.forEach(ky => {
            formattedDataObj[ky][key] = val[ky]
        });

     });

     console.log('formattedData 22 ===> ', JSON.parse(JSON.stringify(formattedDataObj)))

     const formattedData = [];
     Object.keys(formattedDataObj).forEach(key => {
        const val = formattedDataObj[key];
        formattedData.push(val);
     });

     formattedData.sort(function(a,b) { return b.citations - a.citations });

     console.log('formattedData ==>', formattedData)

     $.each(formattedData, function(i, value) {
        $('#top-patents')
        .find('tbody')
        .append('<tr>')
        .append('<td>' + (i + 1) + '</td>')
        .append('<td>' + value.titles + '</td>')
        .append('<td>' + value.assignees + '</td>')
        .append('<td>' + value.application_dates + '</td>')
        .append('<td>' + value.citations + '</td>')
        .append('</tr>');
    });
}

function topPapers(data){
     titles = JSON.parse(data.result.topPaperTable.titles)

     const formattedDataObj = {};
     let dKeys = [];

        dKeys = Object.keys(titles);
        dKeys.forEach(ky => {
            formattedDataObj[ky]= {};
        });

     Object.keys(data.result.topPaperTable).forEach((key, i) => {
        const val = JSON.parse(data.result.topPaperTable[key]);
        console.log(key, val);

        dKeys.forEach(ky => {
            formattedDataObj[ky][key] = val[ky]
        });

     });

     console.log('formattedData 22 ===> ', JSON.parse(JSON.stringify(formattedDataObj)))

     const formattedData = [];
     Object.keys(formattedDataObj).forEach(key => {
        const val = formattedDataObj[key];
        formattedData.push(val);
     });

     formattedData.sort(function(a,b) { return b.citations - a.citations });

     console.log('formattedData ==>', formattedData)

     $.each(formattedData, function(i, value) {
        $('#top-papers')
        .find('tbody')
        .append('<tr>')
        .append('<td>' + (i + 1) + '</td>')
        .append('<td>' + value.titles + '</td>')
        .append('<td>' + value.authors.split(",").join("<br />") + '</td>')
        .append('<td>' + value.affiliations.split(",").join("<br />") + '</td>')
        .append('<td>' + value.years + '</td>')
        .append('<td>' + value.citations + '</td>')
        .append('</tr>');
    });
}

function topPatentInstitutions(data){
    const formattedData = [];
    const formattedDataObj = data.result.topInstitutionsPatents
     Object.keys(formattedDataObj).forEach(key => {
        const val = formattedDataObj[key];
        formattedData.push(val);
     });
      console.log('formattedDatainstitutions ==>', formattedData)
       var r = formattedData[0].map(function(col, i) {
          return formattedData.map(function(row) {
            return row[i];
          });
        });
      r.forEach(function(e,i) {
        $('#institutions-patents')
        .find('tbody')
        .append('<tr>')
        .append('<td>' + (i + 1) + '</td>')
        .append('<td>' + e[0]+ '</td>')
        .append('<td>' + e[1] + '</td>')
        .append('</tr>');
    });
}