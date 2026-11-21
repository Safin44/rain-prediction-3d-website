// ATMOS 3D — light Three.js sky background (unlike the churn site's dark particle field):
// falling rain streaks + soft drifting cloud blobs (light fog).
// Degrades gracefully — no WebGL = CSS gradient fallback, page still works.

(function sky() {
  const canvas = document.getElementById("sky3d");
  if (!canvas || !window.THREE) return;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true }); }
  catch (e) { return; }
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xf3f7f4, 18, 42);
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
  camera.position.set(0, 0, 16);

  // Rain: 900 short streaks falling and recycling.
  const COUNT = 900;
  const pos = new Float32Array(COUNT * 2 * 3);
  const drops = [];
  for (let i = 0; i < COUNT; i++) {
    const x = (Math.random() - 0.5) * 36, y = Math.random() * 22 - 4, z = -Math.random() * 10;
    drops.push({ x, y, z, s: 0.25 + Math.random() * 0.35 });
    pos.set([x, y, z, x + 0.06, y - 0.55, z], i * 6);
  }
  const rainGeo = new THREE.BufferGeometry();
  rainGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const rain = new THREE.LineSegments(rainGeo,
    new THREE.LineBasicMaterial({ color: 0x2f6f9f, transparent: true, opacity: 0.35 }));
  scene.add(rain);

  // Clouds: clusters of soft white spheres drifting sideways.
  const cloudMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
  const clouds = [];
  for (let c = 0; c < 5; c++) {
    const g = new THREE.Group();
    const n = 4 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.9 + Math.random() * 0.9, 18, 14), cloudMat);
      s.position.set(i * 1.1 - n * 0.5, Math.random() * 0.4, Math.random() * 0.5);
      s.scale.y = 0.55;
      g.add(s);
    }
    g.position.set((Math.random() - 0.5) * 26, 5.5 + Math.random() * 3.5, -4 - Math.random() * 4);
    clouds.push({ g, v: 0.004 + Math.random() * 0.006 });
    scene.add(g);
  }
  scene.add(new THREE.AmbientLight(0xffffff, 0.95));
  const sun = new THREE.DirectionalLight(0xffe6a3, 0.7);
  sun.position.set(6, 10, 8); scene.add(sun);

  function resize() {
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  }
  addEventListener("resize", resize); resize();

  (function tick() {
    requestAnimationFrame(tick);
    const p = rainGeo.attributes.position.array;
    for (let i = 0; i < COUNT; i++) {
      const d = drops[i]; d.y -= d.s;
      if (d.y < -6) { d.y = 16; d.x = (Math.random() - 0.5) * 36; }
      p.set([d.x, d.y, d.z, d.x + 0.06, d.y - 0.55, d.z], i * 6);
    }
    rainGeo.attributes.position.needsUpdate = true;
    for (const c of clouds) { c.g.position.x += c.v; if (c.g.position.x > 20) c.g.position.x = -20; }
    renderer.render(scene, camera);
  })();
})();
