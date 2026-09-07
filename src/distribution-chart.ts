import { DENSITY_AXIS_MAX, SPEED_AXIS_MAX, SPEED_BINS, maxwellBoltzmann, mostProbableSpeed } from './speed-distribution';

const NS = 'http://www.w3.org/2000/svg';
const LEFT = 31, TOP = 3, WIDTH = 300, HEIGHT = 45, BASELINE = TOP + HEIGHT;

/** Fixed axes keep changes in speed and peak density visible as temperature changes. */
export function createDistributionChart(container: HTMLElement) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 340 64');
  svg.setAttribute('aria-hidden', 'true');
  const element = <K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string>) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    svg.append(node);
    return node;
  };
  element('path', { d: `M${LEFT} ${TOP}V${BASELINE}H${LEFT + WIDTH}`, class: 'distribution-axis' });
  const label = (text: string, x: number, y: number, anchor = 'middle') => {
    element('text', { x: String(x), y: String(y), 'text-anchor': anchor }).textContent = text;
  };
  label('0.26', LEFT - 4, TOP + 6, 'end');
  label('ps/Å', LEFT - 4, TOP + 20, 'end');
  label('0', LEFT - 4, BASELINE + 2, 'end');
  for (const value of [0, 6, 12, 18]) label(String(value), LEFT + value / SPEED_AXIS_MAX * WIDTH, 60);
  label('Å/ps', LEFT + WIDTH / 2, 60);
  const bars = Array.from({ length: SPEED_BINS }, (_, index) => element('rect', {
    x: String(LEFT + index * WIDTH / SPEED_BINS + 1), y: String(BASELINE),
    width: String(WIDTH / SPEED_BINS - 2), height: '0', class: 'distribution-bar',
  }));
  const curve = element('path', { d: '', class: 'distribution-curve' });
  container.setAttribute('role', 'img');
  container.setAttribute('aria-label', '並進速度分布：計測待ち');
  container.append(svg);
  const y = (density: number) => BASELINE - Math.min(DENSITY_AXIS_MAX, Math.max(0, density)) / DENSITY_AXIS_MAX * HEIGHT;

  return {
    update(bins: ArrayLike<number>, temperature: number) {
      if (bins.length !== SPEED_BINS || !Number.isFinite(temperature) || temperature <= 0) return;
      for (let index = 0; index < SPEED_BINS; index++) if (!Number.isFinite(bins[index])) return;
      bars.forEach((bar, index) => {
        const top = y(bins[index]);
        bar.setAttribute('y', top.toFixed(3));
        bar.setAttribute('height', (BASELINE - top).toFixed(3));
      });
      const path = Array.from({ length: 121 }, (_, index) => {
        const speed = index / 120 * SPEED_AXIS_MAX;
        return `${index === 0 ? 'M' : 'L'}${(LEFT + index / 120 * WIDTH).toFixed(3)},${y(maxwellBoltzmann(speed, temperature)).toFixed(3)}`;
      });
      curve.setAttribute('d', path.join(' '));
      container.setAttribute('aria-label', `分子の並進速度分布。実測温度 ${Math.round(temperature)} K、Maxwell–Boltzmann分布の最確速度 ${mostProbableSpeed(temperature).toFixed(2)} Å/ps。横軸 0–18 Å/ps、確率密度 0–0.26 ps/Å。`);
    },
  };
}
