import { describe, expect, it } from "vitest";
import {
  age,
  duration,
  firstLine,
  humanize,
  integer,
  num,
  percent,
  queryString,
  relativeTime,
  timestamp,
  truncateMiddle,
} from "../src/lib/format";

describe("format", () => {
  it("renders scores with a fixed precision and an em dash for absent values", () => {
    expect(num(0.699178)).toBe("0.699");
    expect(num(null)).toBe("—");
    expect(num(undefined)).toBe("—");
    expect(num(0.5, 2)).toBe("0.50");
  });

  it("renders percentages from 0..1 ratios", () => {
    expect(percent(0.7931034482758621)).toBe("79.3%");
    expect(percent(1)).toBe("100.0%");
    expect(percent(null)).toBe("—");
  });

  it("renders integers with thousands separators", () => {
    expect(integer(12345)).toBe("12,345");
    expect(integer(0)).toBe("0");
    expect(integer(null)).toBe("—");
  });

  it("renders measured durations without inventing a benchmark", () => {
    expect(duration(0.1)).toBe("0.10 ms");
    expect(duration(32.107)).toBe("32.1 ms");
    expect(duration(1500)).toBe("1.50 s");
    expect(duration(null)).toBe("—");
  });

  it("renders ISO timestamps and age in days", () => {
    expect(timestamp("2026-10-14T07:04:57.062175Z")).toContain("2026");
    expect(timestamp("not-a-date")).toBe("not-a-date");
    expect(timestamp(null)).toBe("—");
    expect(age(0.5)).toBe("12.0 h");
    expect(age(5)).toBe("5.0 d");
    expect(age(90)).toBe("3.0 mo");
    expect(age(null)).toBe("—");
  });

  it("renders a relative age", () => {
    const now = new Date().toISOString();
    expect(relativeTime(now)).toBe("just now");
    expect(relativeTime(null)).toBe("—");
  });

  it("truncates in the middle so both ends stay recognisable", () => {
    expect(truncateMiddle("abcdefghij", 9)).toBe("abcd…ghij");
    expect(truncateMiddle("abc", 7)).toBe("abc");
  });

  it("takes the first non-empty line", () => {
    expect(firstLine("\n\nhello\nworld")).toBe("hello");
    expect(firstLine("   ")).toBe("");
  });

  it("humanises identifiers", () => {
    expect(humanize("retrieval_complete")).toBe("Retrieval complete");
    expect(humanize("memory.write")).toBe("Memory write");
    expect(humanize("below_minimum_score")).toBe("Below minimum score");
  });

  it("builds a query string, dropping empty values", () => {
    expect(queryString({ limit: 25, offset: 0, status: undefined, kind: "" })).toBe(
      "?limit=25&offset=0",
    );
    expect(queryString({})).toBe("");
  });
});
