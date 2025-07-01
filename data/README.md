This directory will be used to store spotify streaming data. They will be ignored by git. 

To add data, request the "extended streaming history" from Spotify, unpack the received ZIP, and put all files into the /in folder (so that you end up with a bunch of Streaming_History_Audio_YYYY_I files in the /in folder.).

The /audio-features folder is used to store requested data from a third party audio features API. Don't worry about this folder.

The /out folder is used to store modeled & enriched streaming data for later use of playlist generation