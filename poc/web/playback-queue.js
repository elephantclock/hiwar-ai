export class PlaybackQueue {
  constructor(context) {
    this.context = context;
    this.nextStart = 0;
    this.epoch = 0;
    this.sources = new Set();
  }

  async enqueue(wavBytes, epoch) {
    if (epoch < this.epoch) return;
    if (epoch > this.epoch) this.cancel(epoch);
    const decoded = await this.context.decodeAudioData(wavBytes.slice(0));
    if (epoch !== this.epoch) return;

    const source = this.context.createBufferSource();
    source.buffer = decoded;
    source.connect(this.context.destination);
    source.onended = () => this.sources.delete(source);

    const startAt = Math.max(this.context.currentTime, this.nextStart);
    this.nextStart = startAt + decoded.duration;
    this.sources.add(source);
    source.start(startAt);
  }

  cancel(nextEpoch = this.epoch + 1) {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // A source that already ended is harmless.
      }
    }
    this.sources.clear();
    this.nextStart = this.context.currentTime;
    this.epoch = nextEpoch;
  }
}
