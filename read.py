from tts_pipeline_free import TTSPipeline

pipe = TTSPipeline(
    reference_audio=["./lossless.mp4"],
    temperature=0.9,        # slightly more stable than 0.75 default
    repetition_penalty=7.0, # bump up if you hear stammering
    top_p=0.85,
    speed=0.99,             # a hair slower often sounds more human
)
pipe.speak_text("'Hell is full of good intentions or desires', is a historic aphorism also commonly cited these days as 'The road to hell is paved with good intentions'.The origin of this infamous proverb is attributed to Saint Bernard, a 12th-century French abbot.", output_path="cloned.wav")