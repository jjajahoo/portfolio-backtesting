// 백테스트 계산·숫자 표시·맞춤 설명 (backtester/engine.py, formatting.py, glossary.py와 같은 계산).
// 브라우저에서는 전역 Engine으로, Node(테스트)에서는 require로 쓴다.
const Engine = (() => {
  const DAY = 86400000;
  const days = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / DAY);

  // ---------------------------------------------------------------- 계산
  const periodKey = {
    monthly: (d) => d.slice(0, 7),
    quarterly: (d) => d.slice(0, 4) + "Q" + Math.floor((+d.slice(5, 7) - 1) / 3),
    yearly: (d) => d.slice(0, 4),
  };

  // 새 달/분기/해가 시작된 뒤 첫 거래일의 위치. 첫날은 처음 매수하는 날이라 제외.
  function rebalancePositions(dates, rebalance) {
    if (rebalance === "none") return [];
    const key = periodKey[rebalance];
    if (!key) throw new Error("알 수 없는 리밸런싱 주기: " + rebalance);
    const out = [];
    for (let i = 1; i < dates.length; i++) if (key(dates[i]) !== key(dates[i - 1])) out.push(i);
    return out;
  }

  // prices: {종목: 배열}, weights: {종목: 비중(합 1)}, feeRate: 사고판 금액에 붙는 비용 비율
  function runBacktest(prices, dates, weights, rebalance = "yearly", initial = 10000000, feeRate = 0) {
    const tickers = Object.keys(weights);
    const w = tickers.map((t) => weights[t]);
    const px = tickers.map((t) => prices[t]);
    const n = dates.length;
    const values = new Float64Array(n);
    let units = w.map((wi, k) => (initial * (1 - feeRate) * wi) / px[k][0]);
    let start = 0;
    for (const pos of [...rebalancePositions(dates, rebalance), n]) {
      for (let i = start; i < pos; i++) {
        let v = 0;
        for (let k = 0; k < units.length; k++) v += units[k] * px[k][i];
        values[i] = v;
      }
      if (pos === n) break;
      const holdings = units.map((u, k) => u * px[k][pos]);
      const total = holdings.reduce((a, b) => a + b, 0);
      const traded = holdings.reduce((a, h, k) => a + Math.abs(total * w[k] - h), 0);
      units = w.map((wi, k) => ((total - traded * feeRate) * wi) / px[k][pos]);
      start = pos;
    }
    return values;
  }

  // 각 날짜에 직전 최고점 대비 몇 % 빠져 있는지 (0 이하)
  function drawdown(values) {
    const out = new Float64Array(values.length);
    let peak = -Infinity;
    for (let i = 0; i < values.length; i++) {
      peak = Math.max(peak, values[i]);
      out[i] = values[i] / peak - 1;
    }
    return out;
  }

  // 고점 -> 바닥 -> 회복 구간들. 위치(인덱스)로 돌려준다. recovery가 null이면 아직 회복 전.
  function drawdownPeriods(values) {
    const dd = drawdown(values);
    const out = [];
    let peak = null, trough = null;
    for (let i = 0; i < dd.length; i++) {
      if (dd[i] < 0 && peak === null) {
        peak = i - 1;
        trough = i;
      } else if (peak !== null && dd[i] < 0) {
        if (dd[i] < dd[trough]) trough = i;
      } else if (peak !== null && dd[i] >= 0) {
        out.push({ peak, trough, recovery: i, depth: dd[trough] });
        peak = null;
      }
    }
    if (peak !== null) out.push({ peak, trough, recovery: null, depth: dd[trough] });
    return out;
  }

  // 기간별 수익률. 첫 기간은 시작일 가치에서, 마지막 기간은 마지막 날까지 (일부 기간 포함).
  function periodReturns(values, dates, unit) {
    const key = unit === "year" ? (d) => d.slice(0, 4) : (d) => d.slice(0, 7);
    const out = [];
    let prev = values[0];
    for (let i = 0; i < dates.length; i++) {
      if (i === dates.length - 1 || key(dates[i + 1]) !== key(dates[i])) {
        out.push({ key: key(dates[i]), end: i, ret: values[i] / prev - 1 });
        prev = values[i];
      }
    }
    return out;
  }

  function computeMetrics(values, dates, riskFree = 0, invested = null) {
    const n = values.length;
    const start = dates[0], end = dates[n - 1];
    const initial = invested ?? values[0];
    const final = values[n - 1];
    const years = days(start, end) / 365.25;
    const totalReturn = final / initial - 1;
    const cagr = years > 0 ? Math.pow(final / initial, 1 / years) - 1 : 0;

    // 변동성·샤프 비율은 월 수익률로 (한국/미국 휴장일 차이로 일별 값이 왜곡되기 쉽다)
    const monthly = periodReturns(values, dates, "month").map((p) => p.ret);
    let volatility = null, sharpe = null;
    if (monthly.length >= 3) {
      const mean = monthly.reduce((a, b) => a + b, 0) / monthly.length;
      const variance = monthly.reduce((a, r) => a + (r - mean) ** 2, 0) / (monthly.length - 1);
      volatility = Math.sqrt(variance) * Math.sqrt(12);
      const rfMonthly = Math.pow(1 + riskFree, 1 / 12) - 1;
      if (volatility > 0) sharpe = ((mean - rfMonthly) * 12) / volatility;
    }

    const periods = drawdownPeriods(values);
    const spanDays = (p) => days(dates[p.peak], dates[p.recovery ?? n - 1]);
    let deepest = null, longest = null;
    for (const p of periods) {
      if (!deepest || p.depth < deepest.depth) deepest = p;
      if (!longest || spanDays(p) > spanDays(longest)) longest = p;
    }

    const yearly = periodReturns(values, dates, "year");
    let best = null, worst = null;
    for (const y of yearly) {
      if (!best || y.ret > best.ret) best = y;
      if (!worst || y.ret < worst.ret) worst = y;
    }

    return {
      start, end, initial, final, totalReturn, cagr, volatility, sharpe,
      maxDrawdown: deepest ? deepest.depth : 0,
      maxDrawdownPeriod: deepest && { ...deepest, days: spanDays(deepest) },
      longestDrawdown: longest && { ...longest, days: spanDays(longest) },
      bestYear: best && [+best.key, best.ret],
      worstYear: worst && [+worst.key, worst.ret],
      yearly,
    };
  }

  // ---------------------------------------------------------------- 숫자 표시
  // 원화는 '1억 2,345만 원', 달러는 '$12,345'
  function money(value, currency) {
    if (currency === "USD") return (value < 0 ? "-$" : "$") + Math.round(Math.abs(value)).toLocaleString("en-US");
    const sign = value < 0 ? "-" : "";
    const won = Math.round(Math.abs(value));
    if (won < 10000) return `${sign}${won.toLocaleString("en-US")}원`;
    const total = Math.round(won / 10000);
    const eok = Math.floor(total / 10000), man = total % 10000;
    if (eok && man) return `${sign}${eok.toLocaleString("en-US")}억 ${man.toLocaleString("en-US")}만 원`;
    if (eok) return `${sign}${eok.toLocaleString("en-US")}억 원`;
    return `${sign}${man.toLocaleString("en-US")}만 원`;
  }
  const fixed = (x, digits) => (Object.is(+x.toFixed(digits), -0) ? (0).toFixed(digits) : x.toFixed(digits));
  const pct = (v, digits = 1, signed = false) =>
    v === null || v === undefined ? "-" : (signed && v >= 0 ? "+" : "") + fixed(v * 100, digits) + "%";
  const pp = (v, digits = 1) => (v >= 0 ? "+" : "") + fixed(v * 100, digits) + "%p";
  const ratio = (v) => (v === null || v === undefined ? "-" : fixed(v, 2));
  function duration(d) {
    const months = Math.round(d / 30.44);
    if (months < 1) return `${d}일`;
    const y = Math.floor(months / 12), m = months % 12;
    if (y && m) return `${y}년 ${m}개월`;
    if (y) return `${y}년`;
    return `${m}개월`;
  }
  const ym = (d) => `${+d.slice(0, 4)}년 ${+d.slice(5, 7)}월`;

  // ---------------------------------------------------------------- 맞춤 설명
  function sharpeGrade(s) {
    if (s < 0) return "예금보다 못했어요";
    if (s < 0.5) return "아쉬운 편이에요";
    if (s < 1) return "괜찮은 편이에요";
    return "훌륭한 편이에요 (오랜 기간 1을 넘기긴 어려워요)";
  }

  // 실제 결과 숫자를 넣은 설명 목록. [{title, body(마크다운)}], dates로 위치를 날짜로 바꾼다.
  function explain(m, dates, currency, riskFree, bm = null, benchName = null) {
    const out = [];
    const hasBench = bm && benchName;

    let text =
      `**${ym(m.start)}에 ${money(m.initial, currency)}을 넣고 그대로 뒀다면, ` +
      `${ym(m.end)}에는 ${money(m.final, currency)}이 됐어요.** ` +
      `원금의 ${fixed(m.final / m.initial, 1)}배예요 (총 수익률 ${pct(m.totalReturn, 1, true)}).`;
    if (hasBench) text += `\n\n같은 돈을 ${benchName}에 넣었다면 ${money(bm.final, currency)}이었어요.`;
    out.push({ key: "final", title: "얼마가 됐나요?", body: text });

    text = `매년 평균 **${pct(m.cagr)}씩** 불어난 셈이에요. `;
    text += `해마다 오르내렸지만, '매년 똑같이 ${pct(m.cagr)}씩 올랐다'고 치면 같은 결과가 나와요.`;
    if (m.cagr > 0) {
      const double = Math.log(2) / Math.log(1 + m.cagr);
      text += `\n\n이 속도라면 돈이 **2배가 되는 데 약 ${fixed(double, 0)}년** 걸려요.`;
    }
    text += ` 예금 금리 ${pct(riskFree)}와 비교해 보세요.`;
    if (hasBench) {
      const diff = m.cagr - bm.cagr;
      const word = diff >= 0 ? "높았어요" : "낮았어요";
      text += `\n\n${benchName}의 연평균 수익률 ${pct(bm.cagr)}보다 1년에 ${fixed(Math.abs(diff) * 100, 1)}%p ${word}.`;
    }
    out.push({ key: "cagr", title: "연평균 수익률 (CAGR)", body: text });

    const p = m.maxDrawdownPeriod;
    if (p) {
      const example = currency === "KRW" ? 10000000 : 10000;
      text =
        `가장 힘들었던 때는 **${ym(dates[p.peak])} ~ ${ym(dates[p.trough])}**이에요. ` +
        `고점보다 **${pct(m.maxDrawdown)}까지** 떨어졌어요. ` +
        `그 고점에 계좌에 ${money(example, currency)}이 있었다면 ` +
        `**${money(example * (1 + m.maxDrawdown), currency)}**까지 줄어든 거예요.`;
      if (p.recovery !== null) {
        text += `\n\n다시 고점을 되찾은 건 ${ym(dates[p.recovery])}, 고점부터 **${duration(p.days)}** 걸렸어요.`;
      } else {
        text += "\n\n그리고 기간 마지막 날까지 아직 그 고점을 되찾지 못했어요.";
      }
      if (hasBench) text += `\n\n${benchName}의 최대 낙폭은 ${pct(bm.maxDrawdown)}였어요.`;
      text += "\n\n💡 이 하락을 겪는 동안 팔지 않고 버틸 수 있을지 스스로에게 물어보세요. 백테스트에서 가장 중요한 숫자예요.";
      out.push({ key: "mdd", title: "최대 낙폭 (MDD)", body: text });
    }

    const q = m.longestDrawdown;
    if (q && p && q.peak !== p.peak) {
      const state = q.recovery !== null ? `${ym(dates[q.recovery])}에 회복했어요` : "아직 회복 중이에요";
      text =
        `낙폭이 가장 깊었던 때와는 별개로, 가장 오래 '물려 있던' 기간은 **${ym(dates[q.peak])}부터 ` +
        `${duration(q.days)}**이에요 (${state}). 그동안 최대 ${pct(q.depth)}까지 빠져 있었어요.`;
      out.push({ key: "recovery", title: "가장 긴 회복 기간", body: text });
    }

    if (m.volatility !== null) {
      const lo = m.cagr - m.volatility, hi = m.cagr + m.volatility;
      text =
        `1년 수익률이 평균에서 보통 **±${pct(m.volatility)}** 정도 출렁였다는 뜻이에요. ` +
        `대략 **3년 중 2년은 ${pct(lo)} ~ ${pct(hi)}** 사이였다고 생각하면 돼요.`;
      if (hasBench && bm.volatility !== null) {
        const word = m.volatility < bm.volatility ? "덜" : "더";
        text += `\n\n${benchName}의 변동성 ${pct(bm.volatility)}보다 ${word} 출렁였어요.`;
      }
      out.push({ key: "volatility", title: "변동성", body: text });
    }

    if (m.sharpe !== null) {
      text =
        `샤프 비율 **${ratio(m.sharpe)}** → ${sharpeGrade(m.sharpe)}\n\n` +
        `변동성(마음고생) 1%를 견딜 때마다 예금(${pct(riskFree)})보다 수익률이 ${ratio(m.sharpe)}%p씩 ` +
        "더 높았다는 뜻이에요.";
      if (hasBench && bm.sharpe !== null) {
        const word = m.sharpe >= bm.sharpe ? "좋았어요" : "나빴어요";
        text += `\n\n${benchName}의 샤프 비율 ${ratio(bm.sharpe)}보다 위험 대비 효율이 ${word}.`;
      }
      out.push({ key: "sharpe", title: "샤프 비율", body: text });
    }

    if (m.bestYear && m.worstYear) {
      text =
        `- 가장 좋았던 해: **${m.bestYear[0]}년 ${pct(m.bestYear[1], 1, true)}**\n` +
        `- 가장 나빴던 해: **${m.worstYear[0]}년 ${pct(m.worstYear[1], 1, true)}**`;
      const edges = new Set([+m.start.slice(0, 4), +m.end.slice(0, 4)]);
      const partial = [...new Set([m.bestYear[0], m.worstYear[0]])].filter((y) => edges.has(y)).sort();
      if (partial.length) text += `\n\n(${partial.map((y) => `${y}년`).join(", ")}은 1년이 다 안 되는 기간이에요.)`;
      out.push({ key: "years", title: "최고의 해 / 최악의 해", body: text });
    }
    return out;
  }

  // ---------------------------------------------------------------- 시세 파일 -> 계산용 가격표
  // data: prices.json, base: 'KRW'|'USD'. 모든 종목(과 필요하면 환율)이 데이터가 있는 날부터 자른다.
  function buildPrices(data, tickers, base, startDate, endDate) {
    const fx = data.series["KRW=X"];
    const series = tickers.map((t) => data.series[t]);
    const needFx = series.some((s) => s.currency !== base);
    const firsts = tickers.map((t, k) => ({ label: t, index: series[k].start }));
    if (needFx) firsts.push({ label: "KRW=X", index: fx.start });
    const dataStart = Math.max(...firsts.map((f) => f.index));
    let from = data.dates.findIndex((d) => d >= startDate);
    let to = data.dates.length - 1;
    while (to >= 0 && data.dates[to] > endDate) to--;
    if (from < 0) from = data.dates.length;
    from = Math.max(from, dataStart);
    if (to - from < 1) return { dates: [], prices: {}, limitedBy: null, fxUsed: needFx };
    const dates = data.dates.slice(from, to + 1);
    const prices = {};
    tickers.forEach((t, k) => {
      const s = series[k];
      const arr = new Float64Array(dates.length);
      for (let i = 0; i < dates.length; i++) {
        let v = s.close[from + i - s.start];
        if (s.currency !== base) {
          const rate = fx.close[from + i - fx.start];
          v = base === "KRW" ? v * rate : v / rate;
        }
        arr[i] = v;
      }
      prices[t] = arr;
    });
    const latest = firsts.reduce((a, b) => (b.index > a.index ? b : a));
    return { dates, prices, limitedBy: latest.index === from ? latest : null, fxUsed: needFx };
  }

  return {
    rebalancePositions, runBacktest, drawdown, drawdownPeriods, periodReturns, computeMetrics,
    money, pct, pp, ratio, duration, ym, sharpeGrade, explain, buildPrices,
  };
})();

if (typeof module !== "undefined") module.exports = Engine;
