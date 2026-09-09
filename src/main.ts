import './style.css';
import './homepage.css';
import { initDiffusion } from './diffusion';

document.getElementById('year')!.textContent = String(new Date().getFullYear());
initDiffusion();
