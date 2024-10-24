// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import Stats from "three/addons/libs/stats.module.js";
import { GUI } from "three/addons/libs/lil-gui.module.min.js";

let json_data = {};

let geo = {
  cent: { x: 0, y: 0, z: 0 },
  max: { x: -Infinity, y: -Infinity, z: -Infinity },
  min: { x: +Infinity, y: +Infinity, z: +Infinity },
  scale: null,
  mats_hash: {},
  mat_names: null,
  sources: null,
  receivers: null,
};

let threejs = {
  scene_container: null,
  camera: null,
  controls: null,
  renderer: null,
  scene: null,
  mat_meshes: {},
  stats: null,
  gui: null,
  default_mesh: null,
  model_loaded: false,
};

let gui_obj = {
  colors: {},
  mat_visible: {},
  opacity: null,
  fov: null,
  dist2target: null,
  side: null,
};

function onWindowResize(obj) {
  let scene_container = obj.scene_container;
  let camera = obj.camera;
  let renderer = obj.renderer;

  if (!scene_container) {
    return;
  }
  console.log("You resized the browser window!");
  camera.aspect = scene_container.clientWidth / scene_container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(scene_container.clientWidth, scene_container.clientHeight);
}

async function createScene(obj) {
  // Get a reference to the container element that will hold our scene
  obj.scene_container = document.querySelector("#scene-container");

  if (!obj.scene) {
    obj.scene = new THREE.Scene();
    obj.scene.background = new THREE.Color("skyblue"); //0x87CEEB
  }
}

async function createRenderer(obj, update) {
  let scene_container = obj.scene_container;

  let renderer = obj.renderer;
  if (!renderer) {
    THREE.ColorManagement.enabled = true;

    // create the renderer (canvas)
    obj.renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer = obj.renderer;

    renderer.setSize(scene_container.clientWidth, scene_container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // add the automatically created <canvas> element to the page
    scene_container.appendChild(renderer.domElement);
  }

  // start the animation loop
  renderer.setAnimationLoop(() => {
    update();
    render(obj);
  });
}

async function createStats(obj, update) {
  let scene_container = obj.scene_container;
  let renderer = obj.renderer;
  let stats = obj.stats;
  if (!renderer) {
    return;
  }

  obj.stats = new Stats();
  stats = obj.stats;
  stats.domElement.style.cssText = "position:absolute;top:0px;right:0px;";
  scene_container.appendChild(stats.domElement);
  // restart the animation loop
  renderer.setAnimationLoop(() => {
    update();
    stats.begin();
    render(obj);
    stats.end();
  });
}

function render(obj) {
  let renderer = obj.renderer;
  let scene = obj.scene;
  let camera = obj.camera;
  renderer.render(scene, camera);
}

function sph2cart(az, el) {
  return {
    x: Math.cos((Math.PI / 180.0) * el) * Math.cos((Math.PI / 180.0) * az),
    y: Math.cos((Math.PI / 180.0) * el) * Math.sin((Math.PI / 180.0) * az),
    z: Math.sin((Math.PI / 180.0) * el),
  };
}

async function loadJSON(event) {
  document.getElementById("filepicker").disabled = true;
  let listing = document.createElement("UL");
  listing.id = "listing";
  let container = document.querySelector("#info-container");
  container.innerHTML = "";
  container.appendChild(listing);

  let file = event.target.files[0];
  let item = document.createElement("li");
  item.innerHTML = "Found " + file.webkitRelativePath;
  listing.appendChild(item);

  //read JSON
  let text = await file.text();

  //parse JSON
  json_data = JSON.parse(text);
  geo.mats_hash = json_data.mats_hash;
  geo.sources = json_data.sources;
  geo.receivers = json_data.receivers;
  geo.mat_names = Object.keys(geo.mats_hash).sort();

  window.addEventListener("resize", onWindowResize(threejs));

  item = document.createElement("li");
  item.innerHTML = "drawing model..";
  listing.appendChild(item);

  await drawModel();
}

async function createModelControls() {
  let camera = threejs.camera;
  let renderer = threejs.renderer;
  let controls = new OrbitControls(camera, renderer.domElement);

  controls.screenSpacePanning = true; //allows pan up down
  controls.target.x = geo.cent.x;
  controls.target.y = geo.cent.y;
  controls.target.z = geo.cent.z;
  controls.update();

  threejs.controls = controls;
}

async function createModelGUI() {
  console.log("GUI");
  let scene_container = threejs.scene_container;
  let mat_meshes = threejs.mat_meshes;

  threejs.gui = new GUI();
  let gui = threejs.gui;

  gui.domElement.style.cssText = "position:absolute;top:0px;left:0px;";
  scene_container.appendChild(gui.domElement);

  //folder for material colors
  let f0 = gui.addFolder("Materials");
  let f0m1 = f0.addFolder("Colours");
  let mats = geo.mat_names;
  for (let i = 0; i < mats.length; i++) {
    let mat = mats[i];
    let mesh = mat_meshes[mat];
    gui_obj.colors[mat] = "#" + mesh.material.color.getHexString();
    f0m1.addColor(gui_obj.colors, mat).onChange(function (value) {
      mesh.material.color.set(value); //important this is THREE format color string
    });
  }
  //folder for material visibility
  let f0m2 = f0.addFolder("Visibility");
  for (let i = 0; i < mats.length; i++) {
    let mat = mats[i];
    let mesh = mat_meshes[mat];
    gui_obj.mat_visible[mat] = mat_meshes[mat].visible;
    f0m2.add(gui_obj.mat_visible, mat).onChange(function (value) {
      mat_meshes[mat].visible = value;
    });
  }
  gui_obj.opacity = mat_meshes[mats[0]].material.opacity; //take first value (all same)
  f0m2.add(gui_obj, "opacity", 0, 1).onChange(function (value) {
    //update all opacities identically
    for (let i = 0; i < mats.length; i++) {
      mat_meshes[mats[i]].material.opacity = value;
    }
  });

  //gui_obj.doublesided = (mat_meshes[mats[0]].material.side == THREE.DoubleSide); //take first value (all same)
  gui_obj.side = 1;
  f0m2
    .add(gui_obj, "side", { Front: 0, Back: 1, Both: 2 })
    .onChange(function (value) {
      //update all opacities identically
      for (let i = 0; i < mats.length; i++) {
        if (value == 0) {
          mat_meshes[mats[i]].material.side = THREE.FrontSide;
        } else if (value == 1) {
          mat_meshes[mats[i]].material.side = THREE.BackSide;
        } else {
          mat_meshes[mats[i]].material.side = THREE.DoubleSide;
        }
      }
    });

  let camera = threejs.camera;
  let controls = threejs.controls;
  console.assert(camera != null);
  console.assert(controls != null);
  gui_obj.fov = camera.fov;
  let f1 = gui.addFolder("Camera");
  f1.add(gui_obj, "fov", 5, 90).onChange(function (value) {
    camera.fov = value;
    camera.updateProjectionMatrix();
  });
  //add fov, update position, etc.
  gui_obj.dist2target = camera.position.clone().sub(controls.target).length();
  f1.add(gui_obj, "dist2target").listen();
  f0.open();
  f0m2.open();

  //add visibility
}

async function createModelLights() {
  let scene = threejs.scene;
  const ambientLight = new THREE.AmbientLight(0xffffff, 1);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 1);
  mainLight.position.set(10, 10, 10);

  scene.add(ambientLight, mainLight);
}

async function createModelMesh() {
  let scene = threejs.scene;
  let mat_meshes = threejs.mat_meshes;

  let mats = geo.mat_names;
  for (let i = 0; i < mats.length; i++) {
    let mat = mats[i];
    console.log("Add " + mat);
    let pts = geo.mats_hash[mat].pts;

    const vertices = [];
    const indices = [];

    for (let j = 0; j < pts.length; j += 1) {
      let x = pts[j][0];
      let y = pts[j][1];
      let z = pts[j][2];
      vertices.push(x, y, z);
      if (x > geo.max.x) {
        geo.max.x = x;
      }
      if (y > geo.max.y) {
        geo.max.y = y;
      }
      if (z > geo.max.z) {
        geo.max.z = z;
      }
      if (x < geo.min.x) {
        geo.min.x = x;
      }
      if (y < geo.min.y) {
        geo.min.y = y;
      }
      if (z < geo.min.z) {
        geo.min.z = z;
      }
    }

    let tris = geo.mats_hash[mat].tris;
    for (let j = 0; j < tris.length; j++) {
      let v1 = tris[j][0];
      let v2 = tris[j][1];
      let v3 = tris[j][2];
      indices.push(v1, v2, v3);
    }

    const mat_geometry = new THREE.BufferGeometry();
    mat_geometry.name = mat;
    mat_geometry.setIndex(indices);
    mat_geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3)
    );
    mat_geometry.computeVertexNormals();

    let material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(
        ...geo.mats_hash[mat].color.map(function (c) {
          return c / 255;
        })
      ).convertSRGBToLinear(),
      transparent: true,
      opacity: 0.95,
      side: THREE.BackSide,
    });

    // create a Mesh containing the geometry and material
    let mesh = new THREE.Mesh(mat_geometry, material);

    mat_meshes[mat] = mesh;
    // add the mesh to the scene
    scene.add(mesh);

    //draw edges
    let edges = new THREE.EdgesGeometry(mat_geometry);
    let line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0xffffff })
    );
    scene.add(line);
    threejs.model_loaded = true;
  }

  geo.cent.x = 0.5 * (geo.max.x + geo.min.x);
  geo.cent.y = 0.5 * (geo.max.y + geo.min.y);
  geo.cent.z = 0.5 * (geo.max.z + geo.min.z);
  geo.scale = Math.sqrt(
    (geo.max.x - geo.min.x) ** 2 +
      (geo.max.y - geo.min.y) ** 2 +
      (geo.max.z - geo.min.z) ** 2
  );
}

function addSphere(radius, pos, color) {
  let scene = threejs.scene;

  const geometry = new THREE.SphereGeometry(radius, 32, 32);
  const material = new THREE.MeshBasicMaterial({ color: color });
  const sphere = new THREE.Mesh(geometry, material);

  sphere.position.set(pos[0], pos[1], pos[2]);
  scene.add(sphere);
}

async function createSourceReceiverSpheres() {
  for (let i = 0; i < geo.sources.length; i++) {
    addSphere(0.05, geo.sources[i].xyz, 0x0000ff);
  }
  for (let i = 0; i < geo.receivers.length; i++) {
    addSphere(0.05, geo.receivers[i].xyz, 0x00ff00);
  }
}

async function createModelCamera() {
  let scene_container = threejs.scene_container;

  const fov = 70; // AKA Field of View
  const aspect = scene_container.clientWidth / scene_container.clientHeight;
  const near = geo.scale / 10; // the near clipping plane
  const far = geo.scale * 10; // the far clipping plane
  threejs.camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
  let camera = threejs.camera;

  let cam_pos = { x: 0, y: 0, z: 0 };

  const cam_dist = geo.scale;
  const cam_pos_az = -37.5;
  const cam_pos_el = 30.0;
  //console.log(geo.cent)

  let cam_xyz = sph2cart(cam_pos_az, cam_pos_el);

  cam_pos.x = geo.cent.x + cam_dist * cam_xyz.x;
  cam_pos.y = geo.cent.y + cam_dist * cam_xyz.y;
  cam_pos.z = geo.cent.z + cam_dist * cam_xyz.z;

  camera.near = near;
  camera.far = far;
  camera.fov = fov;

  camera.up.set(0, 0, 1);
  camera.position.set(cam_pos.x, cam_pos.y, cam_pos.z);
  camera.lookAt(geo.cent.x, geo.cent.y, geo.cent.z);
  camera.updateProjectionMatrix();
}

function update_model() {
  let controls = threejs.controls;
  let camera = threejs.camera;
  gui_obj.dist2target = camera.position.clone().sub(controls.target).length();
}

async function drawModel() {
  let upload_container = document.querySelector("#upload-container");
  threejs.scene_container = document.querySelector("#scene-container");
  let scene_container = threejs.scene_container;
  upload_container.style.display = "none";
  upload_container.style.padding = "10px";
  scene_container.style.top = "0px";

  await createScene(threejs);
  await createModelMesh();
  await createSourceReceiverSpheres();
  await createModelCamera();
  await createModelLights();
  await createRenderer(threejs, update_model); //update
  await createStats(threejs, update_model);
  await createModelControls();
  console.assert(threejs.camera != null);
  console.assert(threejs.controls != null);
  await createModelGUI();
}

window.addEventListener("resize", function () {
  onWindowResize(threejs);
});
document.getElementById("filepicker").addEventListener("change", loadJSON);
