/**
 * Subtitle overlay — renders current narration text at the bottom of the
 * 1920×1080 stage frame. Activated via `?sub=1` URL parameter.
 *
 * Design: semi-transparent dark bar, white text, centered — like a
 * standard video subtitle track.
 */
import "./Subtitle.css";

interface Props {
  text: string;
  visible: boolean;
}

export function Subtitle({ text, visible }: Props) {
  if (!visible || !text.trim()) return null;

  return (
    <div className="subtitle-bar">
      <span className="subtitle-text">{text}</span>
    </div>
  );
}
