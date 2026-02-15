import { Vector2 } from "three";
import { UnrealBloomPass } from "three-stdlib";

export type BloomOptions = {
  intensity?: number;
  threshold?: number;
  smoothing?: number;
};

type ComposerLike = {
  addPass: (pass: any) => void;
  removePass?: (pass: any) => void;
  setSize?: (width: number, height: number) => void;
};

type GraphWithComposer = {
  postProcessingComposer?: () => ComposerLike | undefined;
  renderer?: () => { domElement?: { clientWidth: number; clientHeight: number } } | undefined;
};

const DEFAULT_INTENSITY = 1.1;
const DEFAULT_THRESHOLD = 0.0;
const DEFAULT_SMOOTHING = 0.85;

export function attachBloomPostprocess(graph: GraphWithComposer, options: BloomOptions = {}): () => void {
  const composer = graph.postProcessingComposer?.();
  if (!composer) {
    return () => undefined;
  }

  const renderer = graph.renderer?.();
  const width = renderer?.domElement?.clientWidth ?? window.innerWidth;
  const height = renderer?.domElement?.clientHeight ?? window.innerHeight;

  const bloomPass = new UnrealBloomPass(
    new Vector2(width, height),
    options.intensity ?? DEFAULT_INTENSITY,
    options.smoothing ?? DEFAULT_SMOOTHING,
    options.threshold ?? DEFAULT_THRESHOLD
  );

  composer.addPass(bloomPass);
  composer.setSize?.(width, height);

  const handleResize = () => {
    const nextWidth = renderer?.domElement?.clientWidth ?? window.innerWidth;
    const nextHeight = renderer?.domElement?.clientHeight ?? window.innerHeight;
    composer.setSize?.(nextWidth, nextHeight);
    bloomPass.setSize(nextWidth, nextHeight);
  };

  window.addEventListener("resize", handleResize);

  return () => {
    window.removeEventListener("resize", handleResize);
    if (composer.removePass) {
      composer.removePass(bloomPass);
    }
    bloomPass.dispose();
  };
}
