import textwrap
from src.mle_star_ablation.ast_utils import inject_random_state_into_build_fn


def test_inject_random_state_simple():
    src = textwrap.dedent('''
    def build_full_pipeline():
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        p = PCA(n_components=2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        return Pipeline([('pca', p), ('model', model)])
    ''')

    new_src = inject_random_state_into_build_fn(src)
    # Should add arg random_state into signature
    assert 'def build_full_pipeline(random_state' in new_src
    # Should not contain literal 'random_state=42' inside the function anymore
    assert 'random_state=42' not in new_src
    # Should use the variable random_state
    assert 'random_state, ' in new_src or 'random_state)' in new_src or 'random_state]' in new_src
