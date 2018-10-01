# Original Data

https://www.physionet.org/physiobank/database/mitdb/
https://www.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm


Dans le papier, on prédit également des signaux augmentés. Cela me parait plus intéressant de garder le problème désiquilibré.

# Question? 

# Biosppy

ECG function extract : 

    * ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal. 
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds). MAKE NO SENSE WITH ONE BEAT
    templates : array
        Extracted heartbeat templates. MAKE NO SENSE WITH ONE BEAT
    heart_rate_ts : array
        Heart rate time axis reference (seconds). MAKE NO SENSE WITH ONE BEAT
    heart_rate : array
        Instantaneous heart rate (bpm).MAKE NO SENSE WITH ONE BEAT
