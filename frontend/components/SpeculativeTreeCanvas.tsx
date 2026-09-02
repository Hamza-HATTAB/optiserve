"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  RotateCcw,
  Sparkles,
  Info,
  Play,
  Pause,
  CheckCircle2,
  XCircle,
  Zap,
} from "lucide-react";
import { SpeculativeStep } from "@/types";

interface TreeNodeData {
  id: string;
  name: string;
  tokenId: number;
  role: "root" | "accepted" | "rejected" | "resampled" | "bonus" | "pruned_draft";
  position: [number, number, number];
  parentPos?: [number, number, number];
  draftProb: number;
  targetProb: number;
  ratio: number;
  decisionText: string;
}

interface SpeculativeTreeCanvasProps {
  steps: SpeculativeStep[];
  activeCycleIdx?: number;
  onSelectCycle?: (idx: number) => void;
}

export const SpeculativeTreeCanvas: React.FC<SpeculativeTreeCanvasProps> = ({
  steps,
  activeCycleIdx = 0,
  onSelectCycle,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedNode, setSelectedNode] = useState<TreeNodeData | null>(null);
  const [isAutoRotate, setIsAutoRotate] = useState<boolean>(true);
  const [currentStepIdx, setCurrentStepIdx] = useState<number>(activeCycleIdx);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    setCurrentStepIdx(activeCycleIdx);
  }, [activeCycleIdx]);

  const getCycleTreeData = (stepIndex: number): TreeNodeData[] => {
    const step = steps[stepIndex] || steps[0];
    if (!step) return [];

    const nodes: TreeNodeData[] = [
      {
        id: "root-prompt",
        name: "Prompt Context [x₀]",
        tokenId: 101,
        role: "root",
        position: [-6, 0, 0],
        draftProb: 1.0,
        targetProb: 1.0,
        ratio: 1.0,
        decisionText: "Verified Prompt Prefix (Fixed Conditioning Context)",
      },
    ];

    if (stepIndex === 0) {
      nodes.push(
        {
          id: "draft-1",
          name: "x₁: \" calculate\"",
          tokenId: 1204,
          role: "accepted",
          position: [-2, 0.4, 0.3],
          parentPos: [-6, 0, 0],
          draftProb: 0.88,
          targetProb: 0.92,
          ratio: 1.0,
          decisionText: "Accepted (u=0.42 ≤ 1.000). Qwen-1.5B proposal validated by 7B target.",
        },
        {
          id: "draft-1-alt",
          name: "alt: \" compute\"",
          tokenId: 409,
          role: "pruned_draft",
          position: [-2, -2.2, 1.8],
          parentPos: [-6, 0, 0],
          draftProb: 0.08,
          targetProb: 0.04,
          ratio: 0.5,
          decisionText: "Pruned Draft Candidate (Greedy rank #2 in student beam).",
        },
        {
          id: "draft-2",
          name: "x₂: \" total\"",
          tokenId: 345,
          role: "accepted",
          position: [2, -0.2, 0.5],
          parentPos: [-2, 0.4, 0.3],
          draftProb: 0.74,
          targetProb: 0.81,
          ratio: 1.0,
          decisionText: "Accepted (u=0.58 ≤ 1.000). High target cross-entropy alignment.",
        },
        {
          id: "draft-2-alt",
          name: "alt: \" net\"",
          tokenId: 812,
          role: "pruned_draft",
          position: [2, 2.0, -1.5],
          parentPos: [-2, 0.4, 0.3],
          draftProb: 0.14,
          targetProb: 0.07,
          ratio: 0.5,
          decisionText: "Pruned alternative branch.",
        },
        {
          id: "draft-3",
          name: "x₃: \" daily\"",
          tokenId: 891,
          role: "accepted",
          position: [6, 0.3, 0.2],
          parentPos: [2, -0.2, 0.5],
          draftProb: 0.69,
          targetProb: 0.73,
          ratio: 1.0,
          decisionText: "Accepted (u=0.71 ≤ 1.000). All K=3 draft tokens accepted!",
        },
        {
          id: "bonus-token",
          name: "x₄*: \" revenue\"",
          tokenId: 502,
          role: "bonus",
          position: [10, 0, 0],
          parentPos: [6, 0.3, 0.2],
          draftProb: 0.0,
          targetProb: 0.94,
          ratio: 1.0,
          decisionText: "Bonus Target Forward Pass! 4 tokens emitted in 1 cycle (3.8x speedup).",
        }
      );
    } else if (stepIndex === 1) {
      nodes.push(
        {
          id: "draft-1",
          name: "x₁: \" eggs\"",
          tokenId: 4310,
          role: "accepted",
          position: [-2, 0.2, 0.4],
          parentPos: [-6, 0, 0],
          draftProb: 0.82,
          targetProb: 0.86,
          ratio: 1.0,
          decisionText: "Accepted (u=0.31 ≤ 1.000). Valid student prediction.",
        },
        {
          id: "draft-2-rejected",
          name: "x₂: \" sold\"",
          tokenId: 89,
          role: "rejected",
          position: [2.5, 2.4, 0.8],
          parentPos: [-2, 0.2, 0.4],
          draftProb: 0.31,
          targetProb: 0.08,
          ratio: 0.258,
          decisionText: "REJECTED (u=0.74 > p/q=0.258). Student diverged from target distribution.",
        },
        {
          id: "draft-2-resampled",
          name: "x₂*: \" produced\"",
          tokenId: 94,
          role: "resampled",
          position: [2.5, -1.8, -0.6],
          parentPos: [-2, 0.2, 0.4],
          draftProb: 0.05,
          targetProb: 0.89,
          ratio: 1.0,
          decisionText: "Resampled from (p(x) - q(x))⁺. Strict zero distribution shift guaranteed!",
        },
        {
          id: "draft-3-pruned",
          name: "x₃: [pruned]",
          tokenId: 712,
          role: "pruned_draft",
          position: [7, 3.8, 1.2],
          parentPos: [2.5, 2.4, 0.8],
          draftProb: 0.45,
          targetProb: 0.0,
          ratio: 0.0,
          decisionText: "Pruned due to rejection at position #2. Downstream draft discarded.",
        }
      );
    } else {
      nodes.push(
        {
          id: "draft-1",
          name: "x₁: \" equal\"",
          tokenId: 520,
          role: "accepted",
          position: [-2, -0.2, 0.2],
          parentPos: [-6, 0, 0],
          draftProb: 0.91,
          targetProb: 0.94,
          ratio: 1.0,
          decisionText: "Accepted (u=0.23 ≤ 1.000). High student confidence.",
        },
        {
          id: "draft-2",
          name: "x₂: \" to\"",
          tokenId: 114,
          role: "accepted",
          position: [2, 0.3, -0.2],
          parentPos: [-2, -0.2, 0.2],
          draftProb: 0.85,
          targetProb: 0.89,
          ratio: 1.0,
          decisionText: "Accepted (u=0.51 ≤ 1.000).",
        },
        {
          id: "draft-3",
          name: "x₃: \" $18\"",
          tokenId: 608,
          role: "accepted",
          position: [6, -0.1, 0.3],
          parentPos: [2, 0.3, -0.2],
          draftProb: 0.79,
          targetProb: 0.83,
          ratio: 1.0,
          decisionText: "Accepted (u=0.68 ≤ 1.000). Arithmetic answer verified.",
        },
        {
          id: "bonus-token",
          name: "x₄*: \" answer\"",
          tokenId: 203,
          role: "bonus",
          position: [10, 0, 0],
          parentPos: [6, -0.1, 0.3],
          draftProb: 0.0,
          targetProb: 0.96,
          ratio: 1.0,
          decisionText: "Bonus token validated! Full K=3 acceptance speedup.",
        }
      );
    }

    return nodes;
  };

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return;

    const width = containerRef.current.clientWidth || 800;
    const height = 480;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x040508);
    scene.fog = new THREE.FogExp2(0x040508, 0.025);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 7, 18);

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 35;
    controls.minDistance = 6;
    controls.target.set(2, 0, 0);
    controlsRef.current = controls;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f2fe, 1.8);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8b5cf6, 1.5);
    dirLight2.position.set(-10, -10, -10);
    scene.add(dirLight2);

    const particleCount = 280;
    const particleGeometry = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 50;
      particlePositions[i + 1] = (Math.random() - 0.5) * 25;
      particlePositions[i + 2] = (Math.random() - 0.5) * 40;
    }
    particleGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(particlePositions, 3)
    );
    const particleMaterial = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.12,
      transparent: true,
      opacity: 0.4,
    });
    const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particleSystem);

    const gridHelper = new THREE.GridHelper(40, 40, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -4.5;
    scene.add(gridHelper);

    const treeData = getCycleTreeData(currentStepIdx);
    const interactiveMeshes: THREE.Mesh[] = [];
    const animatedRings: THREE.Mesh[] = [];
    const group = new THREE.Group();

    const firstAccepted = treeData.find((n) => n.role === "accepted") || treeData[0];
    setSelectedNode(firstAccepted);

    treeData.forEach((node) => {
      let nodeColor = 0x00f2fe;
      let nodeRadius = 0.65;
      let emissiveIntensity = 0.8;

      if (node.role === "root") {
        nodeColor = 0x8b5cf6;
        nodeRadius = 0.9;
        emissiveIntensity = 1.2;
      } else if (node.role === "rejected") {
        nodeColor = 0xef4444;
        nodeRadius = 0.6;
        emissiveIntensity = 0.6;
      } else if (node.role === "resampled") {
        nodeColor = 0xf59e0b;
        nodeRadius = 0.75;
        emissiveIntensity = 1.1;
      } else if (node.role === "bonus") {
        nodeColor = 0x10b981;
        nodeRadius = 0.8;
        emissiveIntensity = 1.4;
      } else if (node.role === "pruned_draft") {
        nodeColor = 0x64748b;
        nodeRadius = 0.45;
        emissiveIntensity = 0.2;
      }

      const sphereGeo = new THREE.SphereGeometry(nodeRadius, 32, 32);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: nodeColor,
        emissive: nodeColor,
        emissiveIntensity: emissiveIntensity,
        roughness: 0.2,
        metalness: 0.8,
        wireframe: node.role === "pruned_draft",
      });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      sphereMesh.position.set(...node.position);
      sphereMesh.userData = { nodeData: node };
      interactiveMeshes.push(sphereMesh);
      group.add(sphereMesh);

      if (node.role === "accepted" || node.role === "bonus" || node.role === "root") {
        const ringGeo = new THREE.TorusGeometry(nodeRadius * 1.55, 0.04, 16, 64);
        const ringMat = new THREE.MeshBasicMaterial({
          color: nodeColor,
          transparent: true,
          opacity: 0.5,
        });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.position.set(...node.position);
        ringMesh.rotation.x = Math.PI / 2;
        animatedRings.push(ringMesh);
        group.add(ringMesh);
      }

      if (node.parentPos) {
        const p1 = new THREE.Vector3(...node.parentPos);
        const p2 = new THREE.Vector3(...node.position);
        const distance = p1.distanceTo(p2);
        const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

        const cylinderRadius =
          node.role === "accepted" || node.role === "bonus" ? 0.08 : 0.04;
        const cylinderGeo = new THREE.CylinderGeometry(
          cylinderRadius,
          cylinderRadius,
          distance,
          16
        );

        let lineColor = nodeColor;
        let lineOpacity = 0.8;
        if (node.role === "pruned_draft") {
          lineOpacity = 0.25;
        }

        const cylinderMat = new THREE.MeshStandardMaterial({
          color: lineColor,
          emissive: lineColor,
          emissiveIntensity: 0.7,
          transparent: true,
          opacity: lineOpacity,
        });

        const cylinder = new THREE.Mesh(cylinderGeo, cylinderMat);
        cylinder.position.copy(midpoint);
        cylinder.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          p2.clone().sub(p1).normalize()
        );
        group.add(cylinder);
      }
    });

    scene.add(group);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerDown = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactiveMeshes, false);

      if (intersects.length > 0) {
        const clickedMesh = intersects[0].object as THREE.Mesh;
        if (clickedMesh.userData && clickedMesh.userData.nodeData) {
          setSelectedNode(clickedMesh.userData.nodeData);
        }
      }
    };

    const domElement = renderer.domElement;
    domElement.addEventListener("pointerdown", onPointerDown);

    let animationFrameId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      if (isAutoRotate) {
        group.rotation.y = Math.sin(elapsedTime * 0.15) * 0.25;
      }

      animatedRings.forEach((ring, idx) => {
        const scale = 1 + 0.18 * Math.sin(elapsedTime * 3 + idx);
        ring.scale.set(scale, scale, scale);
      });

      particleSystem.rotation.y = elapsedTime * 0.02;

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      domElement.removeEventListener("pointerdown", onPointerDown);
      cancelAnimationFrame(animationFrameId);
      controls.dispose();
      renderer.dispose();
    };
  }, [currentStepIdx, isAutoRotate]);

  const handleResetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
      controlsRef.current.target.set(2, 0, 0);
    }
  };

  const handleCycleChange = (idx: number) => {
    setCurrentStepIdx(idx);
    if (onSelectCycle) {
      onSelectCycle(idx);
    }
  };

  return (
    <div className="relative rounded-2xl border border-white/10 bg-[#040508] overflow-hidden shadow-2xl">
      <div className="relative z-10 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-dark-950/80 p-4 px-6 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-violet-600 shadow-lg shadow-cyan-500/20">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-sans text-sm font-bold tracking-tight text-white sm:text-base">
                3D Speculative Decoding Verification Tree
              </h3>
              <span className="font-mono text-[11px] font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                K = 3 Branching
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium hidden sm:block">
              Interactive WebGL spatial trajectory · Leviathan rejection sampling &amp; bonus token verification
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center rounded-xl bg-dark-900 border border-white/10 p-1">
            {steps.map((s, idx) => (
              <button
                key={idx}
                onClick={() => handleCycleChange(idx)}
                className={`rounded-lg px-2.5 py-1 text-xs font-mono font-bold transition-all ${
                  currentStepIdx === idx
                    ? "bg-cyan-500 text-slate-950 shadow-md"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                Cycle #{idx + 1}
              </button>
            ))}
          </div>

          <button
            onClick={() => setIsAutoRotate(!isAutoRotate)}
            title={isAutoRotate ? "Pause camera orbit" : "Resume camera orbit"}
            className={`rounded-xl border p-2 text-xs transition-all ${
              isAutoRotate
                ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-300"
                : "border-white/10 bg-dark-900 text-slate-400 hover:text-white"
            }`}
          >
            {isAutoRotate ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>

          <button
            onClick={handleResetCamera}
            title="Reset 3D camera"
            className="rounded-xl border border-white/10 bg-dark-900 p-2 text-xs text-slate-400 hover:text-white hover:bg-white/5 transition-all"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <div ref={containerRef} className="relative h-[480px] w-full cursor-grab active:cursor-grabbing">
        <canvas ref={canvasRef} className="h-full w-full outline-none" />

        <div className="absolute top-4 left-6 z-10 hidden sm:flex flex-col gap-1.5 rounded-xl border border-white/10 bg-dark-950/85 p-3 text-xs backdrop-blur-md shadow-xl font-mono">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-violet-500 ring-2 ring-violet-500/30" />
            <span className="text-slate-300">Prompt Root [x₀]</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 ring-2 ring-cyan-400/30" />
            <span className="text-slate-300">Accepted Draft (u &le; p/q)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-500 ring-2 ring-rose-500/30" />
            <span className="text-slate-300">Rejected Candidate</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400 ring-2 ring-amber-400/30" />
            <span className="text-slate-300">Resampled (p - q)&plus;</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 ring-2 ring-emerald-400/30" />
            <span className="text-slate-300">Bonus Target Token</span>
          </div>
        </div>

        <div className="absolute top-4 right-6 z-10 pointer-events-none rounded-lg bg-dark-950/80 px-2.5 py-1 text-[11px] font-mono text-slate-400 border border-white/5 backdrop-blur-sm">
          Drag to Orbit · Scroll to Zoom
        </div>

        {selectedNode && (
          <div className="absolute bottom-4 left-6 right-6 z-10 rounded-xl border border-white/15 bg-dark-950/90 p-4 backdrop-blur-xl shadow-2xl transition-all">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-2.5 mb-2.5">
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-xs font-bold text-white bg-dark-900 border border-white/10 px-2.5 py-1 rounded-md">
                  Token #{selectedNode.tokenId}
                </span>
                <span className="font-sans font-bold text-sm text-cyan-300">
                  {selectedNode.name}
                </span>
                {selectedNode.role === "accepted" && (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                    <CheckCircle2 className="h-3 w-3" /> VERIFIED ACCEPTED
                  </span>
                )}
                {selectedNode.role === "rejected" && (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 rounded-full">
                    <XCircle className="h-3 w-3" /> PRUNED / REJECTED
                  </span>
                )}
                {selectedNode.role === "resampled" && (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-amber-300 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded-full">
                    <Zap className="h-3 w-3" /> RESAMPLED CORRECTIVE
                  </span>
                )}
                {selectedNode.role === "bonus" && (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-emerald-300 bg-emerald-500/15 border border-emerald-500/40 px-2 py-0.5 rounded-full">
                    <Sparkles className="h-3 w-3" /> BONUS TARGET PASS
                  </span>
                )}
              </div>

              <div className="text-[11px] font-mono text-slate-300 flex items-center gap-1">
                <Info className="h-3.5 w-3.5 text-cyan-400" />
                <span>Invariant: P_target(x) &equiv; P_spec(x)</span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="rounded-lg bg-dark-900/90 border border-white/5 p-2.5">
                <span className="text-slate-400 block text-[11px]">Draft Prob q(x)</span>
                <strong className="text-cyan-400 text-sm font-bold">
                  {(selectedNode.draftProb * 100).toFixed(1)}%
                </strong>
              </div>
              <div className="rounded-lg bg-dark-900/90 border border-white/5 p-2.5">
                <span className="text-slate-400 block text-[11px]">Target Prob p(x)</span>
                <strong className="text-emerald-400 text-sm font-bold">
                  {(selectedNode.targetProb * 100).toFixed(1)}%
                </strong>
              </div>
              <div className="rounded-lg bg-dark-900/90 border border-white/5 p-2.5">
                <span className="text-slate-400 block text-[11px]">Acceptance Ratio &alpha;</span>
                <strong className="text-purple-300 text-sm font-bold">
                  {selectedNode.ratio.toFixed(3)}
                </strong>
              </div>
              <div className="rounded-lg bg-dark-900/90 border border-white/5 p-2.5">
                <span className="text-slate-400 block text-[11px]">Sampling Invariant</span>
                <strong className="text-white text-xs font-semibold">
                  u &le; min(1, p/q)
                </strong>
              </div>
            </div>

            <p className="mt-2.5 text-xs text-slate-300 font-sans leading-relaxed">
              <strong className="text-white">Algorithmic Decision:</strong> {selectedNode.decisionText}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
