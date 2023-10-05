1.) Open your Google Drive and create the following directories/files as follows : 

     /MyDrive/Wav2Lip/  (This folder should only contain your Input Video and Input Audio)

     /MyDrive/Wav2lip/ (This folder should contain the pre-trained model / .pth file )

     /MyDrive/YT_LINKS.txt

2.) If your input video is YT video then :
 
    1.) Paste the url/urls in YT_LINKS.txt
    2.) run 2nd cell to install pytube  
    3.)	run the 4th cell to download the video

    If your input video isn't YT url then : 

    1.) upload the input video to /MyDrive/Wav2Lip/ 
    2.) upload the input audio to /MyDrive/Wav2Lip/ 

3.) Download the pre-trained model from the url :
    
    https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1

4.) Upload the downloaded file(from above url) to /MyDrive/Wav2lip/ , it should look like :

    /MyDrive/Wav2lip/wav2lip_gan.pth

5.) run : !git clone https://github.com/Rudrabha/Wav2Lip.git

6.) Replace the function below with _build_mel_basis() in /content/Wav2Lip/audio.py


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )


7.) In the 7th cell , do the following :- 

   1.) !cp "/content/gdrive/My Drive/Wav2Lip/TechNews_Edit_2.mp4" "/content/gdrive/My Drive/Wav2Lip/output10.wav" sample_data/
      
       replace " TechNews_Edit_2.mp4 " with your own input video file name (make sure there aren't any white spaces in your input file's name)
      
       replace "output.wav" with your input audio's file name (make sure there aren't any white spaces in your input file's name)
  
    2.) repat the steps in 1.) for the following comand :

        !cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "../sample_data/TechNews_Edit_2.mp4" --audio "../sample_data/output10.wav" --pads 0 20 0 0 --resize_factor 2 --nosmooth


8.) run the 7th cell and check /content/Wav2Lip/results for result_video.mp4 for download (This is final output) 





     


