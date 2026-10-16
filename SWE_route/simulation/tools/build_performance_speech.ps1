param(
    [string]$VoiceName = 'Microsoft David Desktop',
    [ValidateRange(-10, 10)]
    [int]$Rate = 0
)

# This is a development-only fixture generator. Runtime playback needs only WAV files.
$ErrorActionPreference = 'Stop'
$performanceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../performances'))
$audioRoot = Join-Path $performanceRoot 'audio'
[System.IO.Directory]::CreateDirectory($audioRoot) | Out-Null
Add-Type -AssemblyName System.Speech

function Read-WaveMetadata {
    param([string]$Path)

    $reader = [System.IO.BinaryReader]::new([System.IO.File]::OpenRead($Path))
    try {
        if ([System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'RIFF') {
            throw "Not a RIFF WAV: $Path"
        }
        $reader.ReadUInt32() | Out-Null
        if ([System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'WAVE') {
            throw "Not a WAVE file: $Path"
        }
        $sampleRate = 0
        $blockAlign = 0
        $dataBytes = 0
        while ($reader.BaseStream.Position + 8 -le $reader.BaseStream.Length) {
            $chunkId = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $chunkSize = $reader.ReadUInt32()
            $nextChunk = $reader.BaseStream.Position + $chunkSize + ($chunkSize % 2)
            if ($chunkId -eq 'fmt ') {
                $formatCode = $reader.ReadUInt16()
                $channels = $reader.ReadUInt16()
                $sampleRate = $reader.ReadUInt32()
                $reader.ReadUInt32() | Out-Null
                $blockAlign = $reader.ReadUInt16()
                $bitsPerSample = $reader.ReadUInt16()
                if ($formatCode -ne 1 -or $channels -ne 1 -or $bitsPerSample -ne 16) {
                    throw "Expected mono 16-bit PCM: $Path"
                }
            } elseif ($chunkId -eq 'data') {
                $dataBytes += $chunkSize
            }
            $reader.BaseStream.Position = $nextChunk
        }
        if ($sampleRate -ne 22050 -or $blockAlign -ne 2 -or $dataBytes -eq 0) {
            throw "Invalid fixture audio format or empty audio: $Path"
        }
        if ($dataBytes % $blockAlign -ne 0) {
            throw "Incomplete PCM frame: $Path"
        }
        $frames = [long]($dataBytes / $blockAlign)
        return [ordered]@{
            duration_s = $frames / [double]$sampleRate
            frames = $frames
            sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    } finally {
        $reader.Dispose()
    }
}

$scenes = @(
    [ordered]@{
        id = 'greeting'
        label = 'Welcome back'
        text = "Oh! You're back. I missed you."
        phrases = @('Oh!', "You're back.", 'I missed you.')
    },
    [ordered]@{
        id = 'question'
        label = 'A closer look'
        text = "What's that? Can I take a closer look?"
        phrases = @("What's that?", 'Can I take a closer look?')
    },
    [ordered]@{
        id = 'reassurance'
        label = 'Work it out together'
        text = "It's okay. We can figure this out together."
        phrases = @("It's okay.", 'We can figure this out together.')
    }
)

$speech = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $speech.SelectVoice($VoiceName)
    $speech.Rate = $Rate
    $speech.Volume = 100
    $voiceInfo = $speech.Voice
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
        22050,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono
    )
    $renderedScenes = @()
    foreach ($scene in $scenes) {
        $renderedPhrases = @()
        for ($phraseIndex = 0; $phraseIndex -lt $scene.phrases.Count; $phraseIndex++) {
            $relativeAudio = 'audio/{0}-{1}.wav' -f $scene.id, ($phraseIndex + 1)
            $audioPath = Join-Path $performanceRoot $relativeAudio
            $speech.SetOutputToWaveFile($audioPath, $format)
            $speech.Speak($scene.phrases[$phraseIndex])
            $speech.SetOutputToNull()
            $metadata = Read-WaveMetadata -Path $audioPath
            $renderedPhrases += [ordered]@{
                text = $scene.phrases[$phraseIndex]
                audio = $relativeAudio
                duration_s = $metadata.duration_s
                frames = $metadata.frames
                sha256 = $metadata.sha256
            }
        }
        $renderedScenes += [ordered]@{
            id = $scene.id
            label = $scene.label
            text = $scene.text
            phrases = $renderedPhrases
        }
    }
    $manifest = [ordered]@{
        schema_version = 1
        voice = [ordered]@{
            provider = 'Microsoft'
            engine = 'Windows System.Speech.Synthesis.SpeechSynthesizer'
            engine_assembly_version = [System.Speech.Synthesis.SpeechSynthesizer].Assembly.GetName().Version.ToString()
            name = $voiceInfo.Name
            label = $voiceInfo.Name + ' (prerecorded)'
            id = $voiceInfo.Id
            culture = $voiceInfo.Culture.Name
            rate = $Rate
            volume = 100
            sample_rate_hz = 22050
            channels = 1
            sample_width_bytes = 2
            mode = 'prerecorded local speech fixtures; fixed voice shared across emotions'
        }
        scenes = $renderedScenes
    }
    $json = ($manifest | ConvertTo-Json -Depth 8) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText(
        (Join-Path $performanceRoot 'demo.json'),
        $json + "`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Output ('Rendered {0} scenes / {1} WAV files with {2}.' -f $renderedScenes.Count, ($renderedScenes.phrases | Measure-Object).Count, $voiceInfo.Name)
} finally {
    $speech.Dispose()
}
