"""apollo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.urls import path
from . import views
from . import secondviews
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path(r'data/', views.data, name='data'),
    path(r'testing_data_upload_view/', views.testing_data_upload_view, name='testing_data_upload_view'),
    path(r'training_data_upload_view/', views.training_data_upload_view, name='training_data_upload_view'),
    path(r'runDocumentClassifierSupervised/', views.runDocumentClassifierSupervised,name='runDocumentClassifierSupervised'),
    path(r'runDocumentClassifierUnsupervised/', views.runDocumentClassifierUnsupervised,name='runDocumentClassifierUnsupervised'),
    path(r'home/', views.home, name='home'),
    path(r'svl/', views.svl, name='svl'),
    path(r'usvl/', views.usvl, name='usvl'),
    path(r'em/', views.em, name='em'),
    path(r'emus/', views.emus, name='emus'),
    path(r'il/', views.il, name='il'),
    path(r'ilu/', views.ilu, name='ilu'),
    path(r'ps/', views.ps, name='ps'),
    path(r'da/', views.da, name='da'),
    path(r'um/', views.um, name='um'),
    path(r'sl/', secondviews.sl, name='sl'),
    path(r'redirectChange/', views.redirectChange, name='redirectChange'),
    path(r'fetch_update/', views.fetch_update, name='fetch_update'),
    path(r'fetch_update_unsupervised/', views.fetch_update_unsupervised, name='fetch_update_unsupervised'),
    path(r'xls_response/', views.xls_response, name='xls_response'),
    path(r'training_data_xls_response/', views.training_data_xls_response, name='training_data_xls_response'),
    path(r'save_both_existing_model/', views.save_both_existing_model, name='save_both_existing_model'),
    path(r'save_both_validation/', views.save_both_validation, name='save_both_validation'),
    path(r'retrieve_existing_Project_name/', views.retrieve_existing_Project_name, name='retrieve_existing_Project_name'),
    path(r'retreieve_Model_for_seleted/', views.retreieve_Model_for_seleted,name='retreieve_Model_for_seleted'),
    path(r'patentScoringData/', views.patentScoringData,name='patentScoringData'),
    path(r'computeSimilarityBetweenSamsungAndNonSamsungPatents/', views.computeSimilarityBetweenSamsungAndNonSamsungPatents,name='computeSimilarityBetweenSamsungAndNonSamsungPatents'),
    path(r'makePredictionsForSupervisedLearning/', views.makePredictionsForSupervisedLearning,name='makePredictionsForSupervisedLearning'),
    path(r'makePredictionsForUnsupervisedLearning/', views.makePredictionsForUnsupervisedLearning,name='makePredictionsForUnsupervisedLearning'),
    path(r'incrementalsupervisedlearning/', views.incrementalsupervisedlearning,name='incrementalsupervisedlearning'),
    path(r'incrementalUnsupervisedLearning/', views.incrementalUnsupervisedLearning,name='incrementalUnsupervisedLearning'),
    path(r'run_IL_trainFromScratchFromGUI/', views.run_IL_trainFromScratchFromGUI,name='run_IL_trainFromScratchFromGUI'),
    path(r'getUserName/', views.getUserName,name='getUserName'),
    path(r'fetch_update_patentscoring/', views.fetch_update_patentscoring, name='fetch_update_patentscoring'),
    path(r'runSupervisedSaving/', views.runSupervisedSaving, name='runSupervisedSaving'),
    path(r'runUnsupervisedSaving/', views.runUnsupervisedSaving, name='runUnsupervisedSaving'),
    path(r'patentScoringGlobals/', views.patentScoringGlobals, name='patentScoringGlobals'),
    path('saveAndContinue/', views.saveAndContinue, name='saveAndContinue'),
    path('customFileUpload/', views.customFileUpload, name='customFileUpload'),
    path('removeData/', views.removeData, name='removeData'),
    path('resetDocument/', views.resetDocument, name='resetDocument'),
    path('dataWhenReload/', views.dataWhenReload, name='dataWhenReload'),
    path('categoryName/', views.categoryName, name='categoryName'),
    path('userRunModelTrack/', views.userRunModelTrack, name='userRunModelTrack'),
    path('userRunModelTrackUSL/', views.userRunModelTrackUSL, name='userRunModelTrackUSL'),
    path('userRunModelTrackEM/', views.userRunModelTrackEM, name='userRunModelTrackEM'),
    path('userRunModelTrackIL/', views.userRunModelTrackIL, name='userRunModelTrackIL'),
    path('userRunModelTrackPS/', views.userRunModelTrackPS, name='userRunModelTrackPS'),
    path('Spotlight_Process_All_Files/', secondviews.Spotlight_Process_All_Files, name='Spotlight_Process_All_Files'),
    path('downloads3file/', views.downloads3file, name='downloads3file'),
]