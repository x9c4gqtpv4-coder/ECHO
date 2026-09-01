# 0.5.2 当前逻辑思维导图

```text
稳定主路（保持不变）
源图 + 本SKU指定场景
└─ A0 人物/背景两区域
   ├─ 冻结参数与算法指纹
   ├─ 服装跟随人物整体色彩
   ├─ 从原图单次渲染
   └─ 待人工复核候选

可选精细层（默认关闭）
└─ B1 Precision
   ├─ ATR18部位定位
   ├─ 人工标签/蒙版授权
   └─ 独立部位候选，不改A0

新增只读观察层（默认不进入SKU批处理）
源图 + 参考图 + 可选可比区域/中性蒙版
└─ C1 Observer
   ├─ 输入蒙版和图像哈希绑定
   ├─ 解码线性sRGB亮度分位点
   ├─ 相对曝光型增益
   ├─ 阴影/中调/高光残差
   ├─ 自动低彩度中性假设
   ├─ 显式中性证据下的表观CCT/mired
   ├─ 3×3空间稳定性诊断
   ├─ 源码/阈值/依赖身份指纹
   └─ JSON报告
      ├─ status=review
      ├─ accepted=false
      ├─ pixel_output_changed=false
      ├─ pixel_authority=none
      └─ A0=not_evaluated_not_vetoed

当前严禁
├─ C1修改像素
├─ C1否决或替换A0
├─ 自动中性假设输出CCT
├─ 将显示增益宣称为拍摄EV
├─ 加入OpenCV/Colour/PyTorch生产依赖
└─ 未经真实SKU验证进入C1 Planner/Renderer
```
