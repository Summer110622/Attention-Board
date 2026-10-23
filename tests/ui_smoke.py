"""Browser smoke test (needs: pip install playwright && playwright install chromium).

Usage: python3 -m http.server 8765 &  python3 tests/ui_smoke.py [base_url]
Network access to huggingface.co is mocked, so this runs offline.
"""
import json
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/index.html"
LLAMA = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
    "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8, "hidden_act": "silu",
    "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "vocab_size": 128256, "max_position_embeddings": 8192,
}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    problems = []
    # the mocked 401 for the gated model legitimately logs a failed resource load
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" and "401" not in m.text else None)
    page.on("pageerror", lambda e: problems.append("PAGEERROR " + str(e)))

    def hf_route(route):
        url = route.request.url
        if url.endswith("Llama-X/resolve/main/config.json"):
            route.fulfill(status=200, content_type="application/json", headers={"access-control-allow-origin": "*"}, body=json.dumps(LLAMA))
        elif "gated" in url:
            route.fulfill(status=401, body="", headers={"access-control-allow-origin": "*"})
        else:
            route.fulfill(status=404, body="", headers={"access-control-allow-origin": "*"})

    page.route("https://huggingface.co/**", hf_route)
    page.goto(BASE)
    page.wait_for_selector(".block")

    # starts with the vanilla template, no stray "null" text
    check(page.locator(".block").count() == 10, "vanilla template renders 10 blocks")
    check("null" not in page.locator(".insp-body").inner_text(), "inspector has no stray 'null' text")

    # add a block by click and by drag & drop
    before = page.locator(".block").count()
    page.get_by_role("button", name="Attention dropout").click()
    check(page.locator(".block").count() == before + 1, "click on a palette chip appends a block")
    page.drag_and_drop(".chip:has-text('Mask')", ".block >> nth=1", target_position={"x": 100, "y": 2})
    check(page.locator(".block").count() == before + 2, "drag & drop from the palette inserts a block")

    # undo / redo
    page.keyboard.press("Control+z")
    check(page.locator(".block").count() == before + 1, "Ctrl+Z undoes")
    page.keyboard.press("Control+y")
    check(page.locator(".block").count() == before + 2, "Ctrl+Y redoes")

    # an out-of-order block is flagged, undoing both restores a valid script
    check(page.locator(".block.invalid").count() >= 1, "block placed before its inputs is flagged invalid")
    check("problem" in page.locator(".status").inner_text(), "status bar reports the problem")
    page.keyboard.press("Control+z")
    page.keyboard.press("Control+z")
    check(page.locator(".block").count() == before, "undo restores the original script")

    # swap the normaliser: sparsemax changes the generated code
    page.locator(".block:has-text('softmax')").first.click()
    page.get_by_role("button", name="sparsemax").click()
    page.get_by_role("tab", name="Code").click()
    check("def sparsemax" in page.locator(".code").inner_text(), "sparsemax helper is generated")

    # preview tab draws a heat map
    page.get_by_role("tab", name="Preview").click()
    page.wait_for_selector("canvas.canvas-heat")
    check(page.locator("canvas.canvas-heat").count() == 1, "preview heat map is drawn")

    # diagram view + PNG export
    page.get_by_role("button", name="Diagram").click()
    page.wait_for_selector(".diagram-card svg")
    with page.expect_download() as dl:
        page.get_by_role("button", name="Download PNG").click()
    png_path = dl.value.path()
    with open(png_path, "rb") as f:
        head = f.read(8)
    check(head == b"\x89PNG\r\n\x1a\n", "PNG export produces a valid PNG file")
    page.get_by_role("button", name="Blocks").first.click()

    # Hugging Face dialog (mocked)
    page.get_by_role("button", name="From Hugging Face").click()
    page.fill("#hf-dialog input[type=text]", "https://huggingface.co/org/Llama-X")
    page.get_by_role("button", name="Fetch config").click()
    page.wait_for_selector("#hf-dialog .fig-box svg")
    text = page.locator("#hf-dialog").inner_text()
    check("GQA" in text and "decoder-only" in text, "HF config is parsed (GQA, decoder-only)")
    with page.expect_download() as dl2:
        page.get_by_role("button", name="Download PNG").last.click()
    with open(dl2.value.path(), "rb") as f:
        check(f.read(4) == b"\x89PNG", "model figure PNG is valid")
    page.fill("#hf-dialog input[type=text]", "org/gated-model")
    page.get_by_role("button", name="Fetch config").click()
    page.wait_for_selector("#hf-dialog .msg-err")
    check("gated" in page.locator("#hf-dialog .msg-err").inner_text(), "gated models show a helpful message")
    page.fill("#hf-dialog input[type=text]", "org/Llama-X")
    page.get_by_role("button", name="Fetch config").click()
    page.wait_for_selector("#hf-dialog .fig-box svg")
    page.get_by_role("button", name="Open attention in editor").click()
    check(page.evaluate("window.__ab.state.program.config.Hkv") == 8 and page.evaluate("window.__ab.state.program.config.H") == 32,
          "attention layer is loaded into the editor with GQA settings")
    check(page.evaluate("window.__ab.analysis.errors.length") == 0, "loaded program has no shape errors")

    # language toggle
    page.get_by_role("button", name="日本語").click()
    check(page.get_by_role("button", name="Hugging Faceから").count() == 1, "UI switches to Japanese")
    page.get_by_role("button", name="English").click()

    check(not problems, "no console errors: " + str(problems))
    browser.close()
print("UI smoke test passed")
