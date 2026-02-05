import numpy as np
import random
import matplotlib.pyplot as plt
import os, time
from tqdm import tqdm

# --- 1. 數值工具 ---
def stable_softmax(x):
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z) + 1e-8)

def gaussian_log_prob(x, mu, sigma):
    """計算高斯分佈的對數機率密度"""
    var = sigma**2
    log_prob = -((x - mu)**2) / (2 * var) - np.log(sigma * np.sqrt(2 * np.pi))
    return log_prob

# --- 2. 混合動作空間的線性 PPO Agent (離散 + 連續) ---
class HybridLinearPPO:
    def __init__(self, n_feat, n_discrete_acts, gamma=0.99, lmbda=0.95, eps=0.2):
        """
        混合動作空間 PPO:
        - 離散動作: 前進/左轉/右轉/施法 (4個動作)
        - 連續動作: 當選擇「前進」時控制速度，當選擇「施法」時控制瞄準偏移
        """
        # 離散 Actor 權重 (選擇哪個動作)
        self.w_actor_discrete = np.random.randn(n_discrete_acts, n_feat) * 0.01
        
        # 連續 Actor 權重 (輸出連續動作的均值 mu)
        self.w_actor_continuous_mu = np.random.randn(n_feat) * 0.01
        
        # 連續動作的標準差 (隨訓練衰減以減少探索)
        self.sigma = 0.6  # 初始探索
        self.sigma_min = 0.15  # 最小值
        self.sigma_decay = 0.9998  # 更慢的衰減
        
        # Critic 權重
        self.w_critic = np.random.randn(n_feat) * 0.01
        
        self.gamma = gamma   # 遠見：折現因子
        self.lmbda = lmbda   # GAE：平滑預期
        self.eps = eps       # PPO Clip
        self.buffer = []

    def get_action(self, obs):
        """
        輸出混合動作:
        - a_discrete: 離散動作索引 (0:前進, 1:左轉, 2:右轉, 3:施法)
        - a_continuous: 連續動作值 (前進時的速度調整 或 施法時的瞄準偏移)
        """
        # 1. 離散動作決策
        logits = np.dot(self.w_actor_discrete, obs)
        probs_discrete = stable_softmax(logits)
        probs_discrete = (probs_discrete + 1e-8) / np.sum(probs_discrete + 1e-8)
        a_discrete = np.random.choice(len(probs_discrete), p=probs_discrete)
        
        # 2. 連續動作決策 (從高斯分佈採樣)
        mu = np.dot(self.w_actor_continuous_mu, obs)
        a_continuous = np.random.normal(mu, self.sigma)
        
        # 3. 價值估計
        v = np.dot(self.w_critic, obs)
        
        return a_discrete, a_continuous, probs_discrete[a_discrete], mu, v

    def store(self, transition):
        self.buffer.append(transition)

    def update(self, lr_a_discrete=0.003, lr_a_continuous=0.002, lr_c=0.007):
        if not self.buffer: 
            return
            
        states, a_discretes, a_continuouses, old_probs_discrete, old_mus, values, rewards = zip(*self.buffer)
        
        # --- 計算 GAE 優勢函數 ---
        advantages = []
        gae = 0
        next_v = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_v - v
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            next_v = v
        
        advs = np.array(advantages)
        returns = advs + np.array(values) # Target V

        # --- 更新權重 ---
        for i in range(len(self.buffer)):
            s = states[i]
            a_d = a_discretes[i]
            a_c = a_continuouses[i]
            target_v = returns[i]
            op_d = old_probs_discrete[i]
            o_mu = old_mus[i]
            adv = advs[i]
            
            # 1. Critic MSE 更新
            critic_grad = (target_v - np.dot(self.w_critic, s)) * s
            self.w_critic += lr_c * critic_grad
            
            # 2. 離散 Actor PPO Clip 更新（正確的 softmax 梯度）
            new_probs_discrete = stable_softmax(np.dot(self.w_actor_discrete, s))
            ratio_discrete = new_probs_discrete[a_d] / (op_d + 1e-8)
            
            clipped_ratio_d = np.clip(ratio_discrete, 1-self.eps, 1+self.eps)
            if (ratio_discrete * adv <= clipped_ratio_d * adv) or (1-self.eps < ratio_discrete < 1+self.eps):
                # ✨ 正確的 softmax 策略梯度
                # ∇ log π(a|s) 對 w_a 的導數是: s · (1 - π(a|s))
                # 對其他動作 w_i 的導數是: -s · π(i|s)
                
                # 更新選中的動作
                grad_selected = adv * s * (1 - new_probs_discrete[a_d])
                self.w_actor_discrete[a_d] += lr_a_discrete * grad_selected
                
                # 更新其他動作（softmax 的交叉影響）
                for i in range(len(self.w_actor_discrete)):
                    if i != a_d:
                        grad_other = -adv * s * new_probs_discrete[i]
                        self.w_actor_discrete[i] += lr_a_discrete * grad_other
            
            # 3. 連續 Actor PPO Clip 更新 (基於高斯策略梯度)
            new_mu = np.dot(self.w_actor_continuous_mu, s)
            
            # 計算新舊策略的機率比
            old_log_prob = gaussian_log_prob(a_c, o_mu, self.sigma)
            new_log_prob = gaussian_log_prob(a_c, new_mu, self.sigma)
            ratio_continuous = np.exp(new_log_prob - old_log_prob)
            
            clipped_ratio_c = np.clip(ratio_continuous, 1-self.eps, 1+self.eps)
            if (ratio_continuous * adv <= clipped_ratio_c * adv) or (1-self.eps < ratio_continuous < 1+self.eps):
                # 高斯策略梯度: ∇log π = (a - μ) / σ² * s
                grad = (a_c - new_mu) / (self.sigma**2) * s
                self.w_actor_continuous_mu += lr_a_continuous * adv * grad
                
        # 探索率衰減（減緩衰減速度）
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
                
        self.buffer = []
    
    def get_all_weights(self):
        """返回所有權重的副本（用於記錄）"""
        return {
            'actor_discrete': self.w_actor_discrete.copy(),      # (4, 15)
            'actor_continuous': self.w_actor_continuous_mu.copy(), # (15,)
            'critic': self.w_critic.copy()                        # (15,)
        }

# --- 3. 2D RPG 環境 (支援混合動作空間) ---
class RPG2DEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_angle = 0.0
        self.monster_pos = np.array([8.0, 8.0])
        self.blood_pos = np.array([2.0, 8.0])
        self.wind_up = 0
        return self._get_obs()

    def _get_obs(self):
        rel_m = (self.monster_pos - self.agent_pos) / self.size
        rel_b = (self.blood_pos - self.agent_pos) / self.size
        
        # 基礎特徵：相對位置
        # 怪物和血包的相對位置
        monster_dx, monster_dy = rel_m[0], rel_m[1]
        blood_dx, blood_dy = rel_b[0], rel_b[1]
        
        # 距離特徵
        dist_to_monster = np.linalg.norm(self.monster_pos - self.agent_pos) / self.size
        dist_to_blood = np.linalg.norm(self.blood_pos - self.agent_pos) / self.size
        
        # 角度特徵：目標相對於玩家朝向的角度
        angle_to_monster = np.arctan2(self.monster_pos[1] - self.agent_pos[1], 
                                       self.monster_pos[0] - self.agent_pos[0])
        angle_to_blood = np.arctan2(self.blood_pos[1] - self.agent_pos[1], 
                                     self.blood_pos[0] - self.agent_pos[0])
        
        # 相對角度差（標準化到 [-π, π]）
        relative_angle_monster = np.arctan2(np.sin(angle_to_monster - self.agent_angle),
                                           np.cos(angle_to_monster - self.agent_angle))
        relative_angle_blood = np.arctan2(np.sin(angle_to_blood - self.agent_angle),
                                         np.cos(angle_to_blood - self.agent_angle))
        
        # 是否在瞄準範圍內（布林轉浮點）
        monster_in_sight = 1.0 if abs(relative_angle_monster) < 0.5 and dist_to_monster < 0.6 else 0.0
        
        # 是否接近邊界
        dist_to_wall = min(self.agent_pos[0], self.agent_pos[1], 
                          self.size - self.agent_pos[0], self.size - self.agent_pos[1]) / self.size
        
        # 玩家朝向編碼（三角函數）
        cos_angle = np.cos(self.agent_angle)
        sin_angle = np.sin(self.agent_angle)
        
        # 施法狀態
        casting_progress = self.wind_up / 4.0
        is_ready_to_cast = 1.0 if self.wind_up == 0 else 0.0
        
        # 組合所有特徵
        return np.array([
            # 相對位置 (4)
            monster_dx, monster_dy, blood_dx, blood_dy,
            # 距離 (2)
            dist_to_monster, dist_to_blood,
            # 相對角度 (2)
            relative_angle_monster, relative_angle_blood,
            # 玩家朝向 (2)
            cos_angle, sin_angle,
            # 瞄準判定 (1)
            monster_in_sight,
            # 邊界距離 (1)
            dist_to_wall,
            # 施法狀態 (2)
            casting_progress, is_ready_to_cast,
            # 偏置項 (1)
            1.0
        ])

    def step(self, a_discrete, a_continuous):
        """
        混合動作執行:
        - a_discrete: 0=前進, 1=左轉, 2=右轉, 3=施法
        - a_continuous: 連續值，只在施法時作為瞄準偏移
        """
        reward = -0.01  # 基礎生存懲罰
        event = ""
        
        # 0: 前進 (固定速度，不受連續動作影響)
        if a_discrete == 0:
            move = np.array([np.cos(self.agent_angle), np.sin(self.agent_angle)]) * 0.6
            old_pos = self.agent_pos.copy()
            self.agent_pos = np.clip(self.agent_pos + move, 0, self.size-1)
            
            # 檢查是否碰牆（位置被 clip 限制了）
            if not np.allclose(old_pos + move, self.agent_pos):
                reward -= 2.0
                event = "HIT WALL!"
            
        # 1: 左轉
        elif a_discrete == 1:
            self.agent_angle += 0.4
            
        # 2: 右轉
        elif a_discrete == 2:
            self.agent_angle -= 0.4
            
        # 3: 施法 (a_continuous 控制瞄準偏移角度)
        elif a_discrete == 3 and self.wind_up == 0:
            self.wind_up = 4
            # 連續動作控制瞄準偏移 (clip 在 ±0.5 弧度內)
            aim_offset = np.clip(a_continuous, -0.5, 0.5)
            self.aim_angle = self.agent_angle + aim_offset
            event = "CASTING..."

        # 觸碰血包判定
        if np.linalg.norm(self.agent_pos - self.blood_pos) < 1.0:
            reward += 25
            event = "EAT BLOOD!"
            self.blood_pos = np.random.uniform(1, 9, 2)

        # 施法完成判定
        if self.wind_up > 0:
            self.wind_up -= 1
            if self.wind_up == 0:
                vec = self.monster_pos - self.agent_pos
                angle_to = np.arctan2(vec[1], vec[0])
                # 使用角度差的標準化計算 (處理 ±π 邊界)
                angle_diff = np.arctan2(np.sin(angle_to - self.aim_angle), 
                                       np.cos(angle_to - self.aim_angle))
                
                if abs(angle_diff) < 0.4 and np.linalg.norm(vec) < 6:
                    reward += 12
                    event = "KILLED MONSTER!"
                    self.monster_pos = np.random.uniform(1, 9, 2)
                else:
                    event = "MISSED..."

        return self._get_obs(), reward, event

# --- 4. 訓練與視覺化流程 ---
env = RPG2DEnv()
agent = HybridLinearPPO(n_feat=15, n_discrete_acts=4)
history = []

# 權重演化歷史記錄（記錄完整權重矩陣）
weight_history = {
    'actor_discrete': [],      # 每個時間點記錄 (4, 15) 矩陣
    'actor_continuous': [],    # 每個時間點記錄 (15,) 向量
    'critic': []               # 每個時間點記錄 (15,) 向量
}

print("正在啟動 2D 訓練場... 混合動作空間 PPO 更新機制已就緒。")
print("離散動作: 前進/左轉/右轉/施法")
print("連續動作: 施法時的瞄準偏移")
print("手工特徵: 15 維 (相對位置、距離、角度、瞄準判定、邊界、施法狀態)")
epochs = 3000

# 記錄頻率：每 100 個 episode 記錄一次（避免記憶體爆炸）
record_interval = 100

for ep in tqdm(range(epochs)):
    obs = env.reset()
    total_r = 0
    for _ in range(50):
        a_d, a_c, prob_d, mu, v = agent.get_action(obs)
        next_obs, r, event = env.step(a_d, a_c)
        agent.store((obs, a_d, a_c, prob_d, mu, v, r))
        obs = next_obs
        total_r += r
        
        # 渲染最後 10 輪
        if ep > epochs-10:
            os.system('cls' if os.name == 'nt' else 'clear')
            action_names = ["MOVE", "LEFT", "RIGHT", "CAST"]
            print(f"Ep: {ep} | Reward: {total_r:.1f} | Sigma: {agent.sigma:.3f}")
            print(f"Action: {action_names[a_d]} | Continuous: {a_c:.2f} | Event: {event}")
            
            # 創建帶牆壁的 canvas
            canvas = np.full((10, 10), ".")
            canvas[int(env.monster_pos[1]), int(env.monster_pos[0])] = "M"
            canvas[int(env.blood_pos[1]), int(env.blood_pos[0])] = "B"
            canvas[int(env.agent_pos[1]), int(env.agent_pos[0])] = "A"
            
            # 繪製帶牆壁的地圖
            print("  " + " ".join(["#"] * 12))  # 上牆
            for i, row in enumerate(canvas):
                print("  # " + " ".join(row) + " #")  # 左右牆
            print("  " + " ".join(["#"] * 12))  # 下牆
            
            time.sleep(0.05)
            
    agent.update()
    history.append(total_r)
    
    # 每 record_interval 個 episode 記錄一次完整權重
    if ep % record_interval == 0:
        weights = agent.get_all_weights()
        weight_history['actor_discrete'].append(weights['actor_discrete'])
        weight_history['actor_continuous'].append(weights['actor_continuous'])
        weight_history['critic'].append(weights['critic'])

# 轉換為 numpy 陣列以便處理
weight_history['actor_discrete'] = np.array(weight_history['actor_discrete'])      # (T, 4, 15)
weight_history['actor_continuous'] = np.array(weight_history['actor_continuous'])  # (T, 15)
weight_history['critic'] = np.array(weight_history['critic'])                      # (T, 15)

# --- 5. 繪製權重熱力圖 ---
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 特徵名稱（用於熱力圖標籤）
feature_names = [
    'M_dx', 'M_dy', 'B_dx', 'B_dy',           # 相對位置
    'D_M', 'D_B',                              # 距離
    'θ_M', 'θ_B',                              # 相對角度
    'cos_θ', 'sin_θ',                          # 玩家朝向
    'Sight',                                   # 瞄準判定
    'D_wall',                                  # 邊界距離
    'Cast%', 'Ready',                          # 施法狀態
    'Bias'                                     # 偏置
]

action_names = ['MOVE', 'LEFT', 'RIGHT', 'CAST']

# === 離散 Actor 權重熱力圖 (前3個動作: MOVE, LEFT, RIGHT 放在第一行) ===
for action_idx in range(3):  # 只處理前3個動作
    ax = fig.add_subplot(gs[0, action_idx])
    weight_evolution = weight_history['actor_discrete'][:, action_idx, :].T  # (15, T)
    im = ax.imshow(weight_evolution, aspect='auto', cmap='RdBu_r', 
                  interpolation='nearest', vmin=-0.5, vmax=0.5)
    ax.set_title(f'Actor Discrete: {action_names[action_idx]}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Feature Index', fontsize=9)
    ax.set_xlabel('Episode / 100', fontsize=9)
    ax.set_yticks(range(15))
    ax.set_yticklabels(feature_names, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# CAST 動作單獨放在第二行
ax = fig.add_subplot(gs[1, 0])
weight_evolution = weight_history['actor_discrete'][:, 3, :].T  # (15, T)
im = ax.imshow(weight_evolution, aspect='auto', cmap='RdBu_r', 
              interpolation='nearest', vmin=-0.5, vmax=0.5)
ax.set_title(f'Actor Discrete: {action_names[3]}', fontweight='bold', fontsize=11)
ax.set_ylabel('Feature Index', fontsize=9)
ax.set_xlabel('Episode / 100', fontsize=9)
ax.set_yticks(range(15))
ax.set_yticklabels(feature_names, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# === 連續 Actor 權重熱力圖 ===
ax = fig.add_subplot(gs[1, 1])
weight_evolution = weight_history['actor_continuous'].T  # (15, T)
im = ax.imshow(weight_evolution, aspect='auto', cmap='RdBu_r', 
              interpolation='nearest', vmin=-0.5, vmax=0.5)
ax.set_title('Actor Continuous (μ)', fontweight='bold', fontsize=11)
ax.set_ylabel('Feature Index', fontsize=9)
ax.set_xlabel('Episode / 100', fontsize=9)
ax.set_yticks(range(15))
ax.set_yticklabels(feature_names, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# === Critic 權重熱力圖 ===
ax = fig.add_subplot(gs[1, 2])
weight_evolution = weight_history['critic'].T  # (15, T)
im = ax.imshow(weight_evolution, aspect='auto', cmap='RdBu_r', 
              interpolation='nearest', vmin=-0.5, vmax=0.5)
ax.set_title('Critic', fontweight='bold', fontsize=11)
ax.set_ylabel('Feature Index', fontsize=9)
ax.set_xlabel('Episode / 100', fontsize=9)
ax.set_yticks(range(15))
ax.set_yticklabels(feature_names, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# === 訓練曲線（底部佔據整行）===
ax = fig.add_subplot(gs[2, :])
ax.plot(history, color='royalblue', alpha=0.3, label="Raw Episode Reward")
moving_avg = np.convolve(history, np.ones(50)/50, mode='valid')
ax.plot(range(len(moving_avg)), moving_avg, color='crimson', linewidth=2, label="50-Ep Rolling Average")
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title("Training Reward Curve", fontsize=12, fontweight='bold')
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("RPG Agent: Complete Weight Evolution Heatmaps (All 90 Parameters)", 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('weight_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n訓練完成！")
print(f"最終 100 輪平均獎勵: {np.mean(history[-100:]):.2f}")
print(f"\n權重統計:")
print(f"  Actor Discrete:   60 個權重 (4 actions × 15 features)")
print(f"  Actor Continuous: 15 個權重")
print(f"  Critic:           15 個權重")
print(f"  總計:             90 個權重參數")