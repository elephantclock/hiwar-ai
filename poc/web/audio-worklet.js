class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate;
    this.source = [];
    this.cursor = 0;
    this.frame = [];
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (const sample of channel) this.source.push(sample);

    while (this.cursor + 1 < this.source.length) {
      const left = Math.floor(this.cursor);
      const fraction = this.cursor - left;
      const value =
        this.source[left] * (1 - fraction) + this.source[left + 1] * fraction;
      const clamped = Math.max(-1, Math.min(1, value));
      this.frame.push(clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff);
      this.cursor += this.ratio;

      if (this.frame.length === 320) {
        const pcm = Int16Array.from(this.frame);
        this.port.postMessage({ type: "pcm", pcm: pcm.buffer }, [pcm.buffer]);
        this.frame = [];
      }
    }

    const consumed = Math.floor(this.cursor);
    if (consumed > 0) {
      this.source = this.source.slice(consumed);
      this.cursor -= consumed;
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
