"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, save_learning_theme

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定
    assert "save_learning_log" not in offered

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_approval_saves_without_another_model_run(monkeypatch, tmp_path):
    import app.agent as agent_module

    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    approval_prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: approval_prompts.append(prompt) or "y")

    class PendingAgent:
        run_count = 0

        def run_sync(self, prompt, *, model):
            self.run_count += 1
            self.prompt = prompt
            call = ToolCallPart(
                "save_learning_log",
                {"theme": "LoRA", "summary": "LoRAは低ランク行列で追加学習する手法。"},
            )
            return type("Result", (), {"output": DeferredToolRequests(approvals=[call])})()

    pending_agent = PendingAgent()
    monkeypatch.setattr(agent_module, "learning_log_agent", pending_agent)

    result = save_learning_theme("LoRAとは？", "LoRAは低ランク行列で追加学習する手法です。", model=None)

    assert pending_agent.run_count == 1
    assert "質問:\nLoRAとは？" in pending_agent.prompt
    assert "回答:\nLoRAは低ランク行列で追加学習する手法です。" in pending_agent.prompt
    assert "LoRAは低ランク行列で追加学習する手法。" in approval_prompts[0]
    assert result == "学習テーマ「LoRA」と回答要約を記録しました。"
    assert (log_dir / "themes.txt").read_text(encoding="utf-8") == (
        "LoRA: LoRAは低ランク行列で追加学習する手法。\n"
    )


def test_main_prints_answer_before_starting_learning_log(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    import app.agent as agent_module

    class AnswerAgent:
        def run_sync(self, question, *, model):
            return SimpleNamespace(output="回答です")

    def save_after_answer(question, answer, model):
        assert "回答です" in capsys.readouterr().out
        return "保存しました"

    monkeypatch.setattr(agent_module, "agent", AnswerAgent())
    monkeypatch.setattr(agent_module, "build_model", lambda: None)
    monkeypatch.setattr(agent_module, "save_learning_theme", save_after_answer)
    monkeypatch.setattr(sys, "argv", ["app.agent", "テスト質問"])

    agent_module.main()
